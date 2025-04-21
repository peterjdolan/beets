# This file is part of beets.
# Copyright 2016, Adrian Sampson.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

"""Simple but robust implementation of generator/coroutine-based
pipelines in Python. The pipelines may be run either sequentially
(single-threaded) or in parallel (one thread per pipeline stage).

This implementation supports pipeline bubbles (indications that the
processing for a certain item should abort). To use them, yield the
BUBBLE constant from any stage coroutine except the last.

In the parallel case, the implementation transparently handles thread
shutdown when the processing is complete and when a stage raises an
exception. KeyboardInterrupts (^C) are also handled.

When running a parallel pipeline, it is also possible to use
multiple coroutines for the same pipeline stage; this lets you speed
up a bottleneck stage by dividing its work among multiple threads.
To do so, pass an iterable of coroutines to the Pipeline constructor
in place of any single coroutine.
"""

from __future__ import annotations

import asyncio
import functools
from typing import AsyncGenerator, Callable, Generator, Generic, Iterable, TypeVar, Literal
from beets.util import parallel, async_state_machine as asm

from typing_extensions import TypeVar

# A special value that a pipeline stage can yield to indicate "do not pass task to next stage."
BUBBLE = Literal["__PIPELINE_BUBBLE__"]

DEFAULT_QUEUE_SIZE = 16


T = TypeVar("T")  # Type of the task
# Normally these are concatenated i.e. (*args, task)


class MultiMessage(Generic[T]):
    """A message yielded by a pipeline stage encapsulating multiple
    values to be sent to the next stage.
    """

    def __init__(self, messages: Iterable[T]):
        self.messages = messages


def multiple(messages: Iterable[T]) -> MultiMessage[T]:
    """Yield multiple([message, ..]) from a pipeline stage to send
    multiple values to the next pipeline stage.
    """
    return MultiMessage(messages)


# Return type of the function (should normally be task but sadly
# we cant enforce this with the current stage functions without
# a refactor)
R = TypeVar("R")
StageFn = Callable[
    [T | None], R | Generator[R, None, None] | MultiMessage[R] | BUBBLE
]
MutatorStageFn = Callable[[T | None], None]
_PipelineStateHandler = Callable[[T | None], Generator[R, None, None]]


def _allmsgs(obj: R | MultiMessage[R] | BUBBLE) -> list[R]:
    """Returns a list of all the messages encapsulated in obj. If obj
    is a MultiMessage, returns its enclosed messages. If obj is BUBBLE,
    returns an empty list. Otherwise, returns a list containing obj.
    """
    if isinstance(obj, MultiMessage):
        return obj.messages
    elif obj == BUBBLE:
        return []
    else:
        return [obj]


def stage(
    func: StageFn | None = None,
    local: bool = False,
) -> _PipelineStateHandler:
    """Decorate a function to become a simple stage.

    >>> @stage
    ... def add(n, i):
    ...     return i + n
    >>> pipe = Pipeline([
    ...     iter([1, 2, 3]),
    ...     add(2),
    ... ])
    >>> list(pipe.pull())
    [3, 4, 5]
    """

    @functools.wraps(func)
    async def handler_fn(task: T) -> AsyncGenerator[R, None]:
        if local:
            vals = func(task)
        else:
            future = parallel.submit(func, task)
            while not future.done():
                await asyncio.sleep(0.01)
            if future.exception():
                print(f"Pipeline stage {func.__name__} raised an exception: {future.exception()}")
                return

            vals = future.result()

        if isinstance(vals, Generator):
            for val in vals:
                yield val
        else:
            for val in _allmsgs(vals):
                yield val

    return handler_fn


def mutator_stage(func: MutatorStageFn) -> _PipelineStateHandler:
    """Decorate a function that manipulates items in a coroutine to
    become a simple stage.

    >>> @mutator_stage
    ... def setkey(key, item):
    ...     item[key] = True
    >>> pipe = Pipeline([
    ...     iter([{'x': False}, {'a': False}]),
    ...     setkey('x'),
    ... ])
    >>> list(pipe.pull())
    [{'x': True}, {'a': False, 'x': True}]
    """

    @functools.wraps(func)
    async def handler_fn(task: T) -> AsyncGenerator[R, None]:
        future = parallel.submit(func, task)
        while not future.done():
            await asyncio.sleep(0.01)
        if future.exception():
            print(f"Pipeline stage {func.__name__} raised an exception: {future.exception()}")
            return

        future.result()
        yield task

    return handler_fn


class Pipeline:
    """Represents a staged pattern of work. Each stage in the pipeline
    is a coroutine that receives messages from the previous stage and
    yields messages to be sent to the next stage.
    """

    def __init__(self, stages: Iterable[_PipelineStateHandler]):
        """Makes a new pipeline from a list of pipeline state handlers.

        There must be at least two stages. The state handlers are
        constructed by decorating stage functions with either @stage or
        @mutator_stage.
        """
        self.stage_handler_fns = tuple(stages)
        if len(self.stage_handler_fns) < 2:
            raise ValueError("pipeline must have at least two stages")

    async def run_sequential(self):
        """Run the pipeline sequentially in the current thread. The
        stages are run one after the other. Only the first coroutine
        in each stage is used.
        """
        async for _ in self.pull():
            pass

    async def run_parallel(self, queue_size=DEFAULT_QUEUE_SIZE):
        """Run the pipeline in parallel using one thread per stage. The
        messages between the stages are stored in queues of the given
        size.
        """
        graph = asm.Graph(
            (
                asm.State(
                    id=f"stage_{i}",
                    handler=stage_handler_fn,
                    user_interaction=False,
                    max_queue_size=queue_size,
                ),
                ((f"stage_{i + 1}", lambda _: True),)
                if i < len(self.stage_handler_fns) - 1
                else tuple(),
            )
            for i, stage_handler_fn in enumerate(self.stage_handler_fns)
        )

        async with asm.AsyncStateMachine(graph) as machine:
            # Hack - we know that the first stage of the pipeline is a producer, which takes
            # no arguments and yields the tasks to be processed.
            await machine.inject(None, "stage_0")
            await machine.join()

    async def pull(self) -> AsyncGenerator[R, None]:
        """Yield elements from the end of the pipeline. Runs the stages
        sequentially until the last yields some messages. Each of the messages
        is then yielded by ``pulled.next()``. If the pipeline has a consumer,
        that is the last stage does not yield any messages, then pull will not
        yield any messages. Only the first coroutine in each stage is used
        """
        handler_fns = self.stage_handler_fns
        async for initial in handler_fns[0]:
            prevs = [initial]
            for fn in handler_fns[1:]:
                next_msgs = []
                for prev in prevs:
                    async for next in fn(prev):
                        next_msgs.append(next)
                prevs = next_msgs
            for prev in prevs:
                yield prev


# Smoke test.
if __name__ == "__main__":
    import time

    # Test a normally-terminating pipeline both in sequence and
    # in parallel.
    @stage(local=True)
    def produce(_: int | None) -> Generator[int, None, None]:
        for i in range(5):
            print("generating", i)
            time.sleep(1)
            yield i

    @stage
    def work(i: int) -> int:
        print("processing", i)
        time.sleep(2)
        return i * 2

    @stage
    def consume(i: int) -> None:
        time.sleep(1)
        print("received", i)

    ts_start = time.time()
    asyncio.run(Pipeline([produce, work, consume]).run_sequential())
    ts_seq = time.time()
    asyncio.run(Pipeline([produce, work, consume]).run_parallel())
    ts_par = time.time()
    print("Sequential time:", ts_seq - ts_start)
    print("Parallel time:", ts_par - ts_seq)
    print()

    # Test a pipeline that raises an exception.
    @stage(local=True)
    def exc_produce(_: int | None) -> Generator[int, None, None]:
        for i in range(10):
            print("generating %i" % i)
            time.sleep(1)
            yield i

    @stage
    def exc_work(num: int) -> int:
        print("processing %i" % num)
        time.sleep(3)
        if num == 3:
            raise Exception()
        return num * 2

    @stage
    def exc_consume(num: int) -> None:
        print("received %i" % num)

    asyncio.run(Pipeline([exc_produce, exc_work, exc_consume]).run_parallel(1))
