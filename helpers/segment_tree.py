import operator


class SegmentTree(object):

    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure (https://en.wikipedia.org/wiki/Segment_tree).
        Can be used as regular array, but with two important differences:
            1) setting item's value is slightly slower (O(log capacity) instead of O(1))
            2) user has access to an efficient `reduce` operation which reduces `operation`
               over a contiguous subsequence of items in the array.

        Args:
            capacity (int): Total size of the array - must be a power of two
            op (lambda obj, obj -> obj): Internal operation of a mathematical group for
                combining elements (e.g. sum, max)
            neutral_element (obj): Neutral element for the operation above.
                (e.g. float('-inf') for max and 0 for sum)
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and \
                                                                 a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Return result of applying `self._operation` to a contiguous subsequence of the array.

        Args:
            start (int): beginning of the subsequences
            end (int): end of the subsequences
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity=capacity,
                                             operation=operator.add,
                                             neutral_element=0.0)

    def sum(self, start=0, end=None):
        """Return arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - 1]) <= prefixsum
        It directly exploits the property that segment trees provide an efficient way of
        calculating cumulative sums of elements. If array values are probabilities, this
        function allows to sample indexes according to the discrete probability efficiently.
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(capacity=capacity,
                                             operation=min,
                                             neutral_element=float('inf'))

    def min(self, start=0, end=None):
        """Return min(arr[start], ...,  arr[end])"""
        return super(MinSegmentTree, self).reduce(start, end)
