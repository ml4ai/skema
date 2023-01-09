"""Wrapper for an async thread operating on data in a queue."""
import queue
import threading


def _thread_target(_queue, func, pre_func, post_func, **kwargs):
    """Wrapper around functionals that becomes the target for the thread."""
    if pre_func is not None:
        pre_func_output = pre_func(**kwargs)
        if pre_func_output is not None:
            kwargs = {**kwargs, **pre_func_output}
    for item in iter(_queue.get, None):
        func_output = func(item, **kwargs)
        if func_output is not None:
            kwargs = {**kwargs, **func_output}
        _queue.task_done()

    if post_func is not None:
        post_func(**kwargs)
    _queue.task_done()


class AsyncQueueThread:
    """Async thread that processes data put into its queue."""

    def __init__(self, func, pre_func=None, post_func=None, queue_maxsize=100, **kwargs):
        """Init TODO describe functionals."""
        self._func = func
        self._pre_func = pre_func
        self._post_func = post_func
        self._queue = queue.Queue(queue_maxsize)
        self._thread = threading.Thread(
            target=_thread_target,
            args=(self._queue, self._func, self._pre_func, self._post_func),
            kwargs=kwargs,
        )
        self._thread.start()

    def put(self, x):
        """Add item to thread's queue."""
        self._queue.put(x)

    def close(self):
        """Close thread and clean up."""
        self._queue.put(None)
        self._thread.join()


# pylint: disable=pointless-string-statement
'''
if __name__ == "__main__":

    def func_(x, asdf, y, **kwargs):  # pylint: disable=unused-argument
        """Test target."""
        print(asdf)
        print(x + y)

    def pre_func_(**kwargs):  # pylint: disable=unused-argument
        """Test pre_func."""
        y = 2
        print(locals())
        return {"y": y}

    test = AsyncQueueThread(func_, pre_func=pre_func_, asdf="asdf")
    for i in range(10):
        test.put(i)

    test.close()
'''
