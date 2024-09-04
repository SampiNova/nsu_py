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
