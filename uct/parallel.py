"""Vague attempt at doing MCTS using a few cores"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import multiprocessing
import time


class Searcher(multiprocessing.Process):
    """Pulls nodes off the task queue, does a few searches on them and
    puts the results back in the other queue."""

    def __init__(self, task_queue, result_queue,
                 tree_depth, search_func):
        """
        A process which sits around until it pulls a job off the queue.
        Does the job, gives the results back in the second queue.

        Args:
            task_queue: a JoinableQueue to pull tasks off.
            result_queue: a queue to put results onto
            tree_depth: how far to expand the tree using some kind of policy
                before switching to uniform rollouts.
            search_func: a function which does the searching
        """
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.tree_depth = tree_depth
        self.do_search = search_func

    def run(self):
        name = self.name
        print('{}: beginning'.format(name))
        while True:
            # TODO: would it be better to get a state or a state, action pair?
            job = self.task_queue.get()
            if job is None:
                self.task_queue.task_done()
                print('{}: exiting'.format(name))
                break
            node, num = job
            # do the work
            q_sum = 0
            print('{}: doing {} searches'.format(name, num))
            for _ in xrange(num):
                q_sum += search_func(node, self.tree_depth)

            time.sleep(0.1)
            self.result_queue.put((node, q_sum))
            self.task_queue.task_done()


class ParallelSearcher(object):
    """Wraps up some parallel searching"""

    def __init__(self, max_depth, search_func,
                 num_processes=multiprocessing.cpu_count()*2):
        """Gets everything good to go.

        Args:
            max_depth: the maximum depth to go down the tree before starting
                a rollout.
            search_func: a functon applied to 
            num_processes: how many searches to do in parallel.
            -- will require more soon.
        """
        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        self.consumers = [Searcher(self.tasks, self.results, max_depth)
                          for _ in xrange(num_processes)]
        print('STARTING SEARCH PROCESSES')
        for proc in self.consumers:
            proc.start()

    def stop(self):
        """Closes all processes"""
        print('KILLING')
        for _ in xrange(len(self.consumers)):
            self.tasks.put(None)
        self.tasks.join()
        print('DONE')

    def search(self, start, game_descr, num=128):
        """Does some parallel searching from the given starting point"""
        for action in game_descr.legal_actions(start):
            self.tasks.put(((start, action), num))
        self.tasks.join()
        # can iter Queue?
        while not self.results.empty():
            print('Result: {}'.format(self.results.get()[1]))

if __name__ == '__main__':
    srcher = ParallelSearcher(5)
    result = srcher.search(0)
    srcher.stop()
    print(result)
