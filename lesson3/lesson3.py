class Item:
    def __init__(self, count=0, max_count=16):
        self._count = count
        self._max_count = max_count

    def update_count(self, val):
        if 0 <= val <= self._max_count:
            self._count = val
            return True
        else:
            return False

    @property
    def count(self):
        return self._count

    def __add__(self, num):
        return self._count + num

    def __sub__(self, num):
        return self._count - num

    def __mul__(self, num):
        return self._count * num

    def __iadd__(self, num):
        self._count += num

    def __isub__(self, num):
        self._count -= num

    def __imul__(self, num):
        self._count *= num

    def __lt__(self, num):
        return self._count < num

    def __gt__(self, num):
        return self._count > num

    def __le__(self, num):
        return self._count <= num

    def __ge__(self, num):
        return self._count >= num

    def __eq__(self, num):
        return self._count == num

    def __len__(self):
        return self._count


class Fruit(Item):
    def __init__(self, ripe=True, **kwargs):
        super().__init__(**kwargs)
        self._ripe = ripe


class Food(Item):
    def __init__(self, saturation, **kwargs):
        super().__init__(**kwargs)
        self._saturation = saturation

    @property
    def eatable(self):
        return self._saturation > 0


class Kiwi(Fruit, Food):
    def __init__(self, meat, ripe, count=1, max_count=50, saturation=5):
        super().__init__(ripe=ripe, count=count, max_count=max_count, saturation=saturation)
        self.meat = meat

    @property
    def meat(self):
        return self.meat

    @property
    def eatable(self):
        return super().eatable and self._ripe
