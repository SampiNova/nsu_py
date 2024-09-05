class Stack(list):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lst = list()

    def push(self, elem):
        self.lst.append(elem)

    def pop(self):
        if bool(self.lst):
            return self.lst.pop(len(self.lst) - 1)
        return None


class Queue(list):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lst = list()

    def push(self, elem):
        self.lst.append(elem)

    def pop(self):
        if bool(self.lst):
            return self.lst.pop(0)
        return None

    def __iter__(self):
        return iter(self.lst)


queue = Queue()
for i in range(10):
    queue.push(i + 1)
temp = list(queue)
print(temp)
temp = temp[:3]
print(temp)
for i in range(10):
    print(queue.pop(), end=' ')
