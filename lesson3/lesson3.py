class Item:
    def __init__(self, count=1, max_count=16):
        self._count = count
        self._max_count = max_count

    def _update_count(self, val):
        if 0 <= val <= self._max_count:
            self._count = val
            return True
        else:
            return False

    def _test(self, val):
        if val < 0:
            return 0
        elif val > self._max_count:
            return self._max_count
        else:
            return val

    @property
    def count(self):
        return self._count

    def __add__(self, num):
        return self._test(self._count + num)

    def __sub__(self, num):
        return self._test(self._count - num)

    def __mul__(self, num):
        return self._test(self._count * num)

    def __iadd__(self, num):
        self._count = self._test(self._count + num)
        return self

    def __isub__(self, num):
        self._count = self._test(self._count - num)
        return self

    def __imul__(self, num):
        self._count = self._test(self._count * num)
        return self

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
        self._meat = meat

    @property
    def meat(self):
        return self._meat

    @property
    def eatable(self):
        return super().eatable and self._ripe

    def copy(self):
        return Kiwi(self._meat, self._ripe)


class Mango(Fruit, Food):
    def __init__(self, color, ripe, count=1, max_count=10, saturation=15):
        super().__init__(ripe=ripe, count=count, max_count=max_count, saturation=saturation)
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return super().eatable and self._ripe

    def copy(self):
        return Mango(self._color, super()._ripe)


class Biscuit(Food):
    def __int__(self, mode, count=5, max_count=100, saturation=3):
        super().__init__(count=count, max_count=max_count, saturation=saturation)
        self._mode = mode

    @property
    def mode(self):
        return self.mode

    @property
    def eatable(self):
        return super().eatable()

    def copy(self):
        return Biscuit(self._mode, count=1)


class Pasta(Food):
    def __int__(self, mode, count=1, max_count=3, saturation=40):
        super().__init__(count=count, max_count=max_count, saturation=saturation)
        self._mode = mode

    @property
    def mode(self):
        return self._mode

    @property
    def eatable(self):
        return super().eatable()

    def copy(self):
        return Pasta(self._mode)


class Inventory:
    def __init__(self, count=1):
        self._count = count
        self.inventory = [None for _ in range(count)]

    @property
    def count(self):
        return self.count

    def get(self, idx, count):
        if self.inventory[idx] is None:
            return None
        else:
            res = self.inventory[idx].copy()
            self.inventory[idx] -= count
            if self.inventory[idx].count <= 0:
                self.inventory[idx] = None
            return res

    def __setitem__(self, idx, value):
        if self.inventory[idx] is None:
            self.inventory[idx] = value
        elif type(self.inventory[idx]) is type(value):
            self.inventory[idx] += value.count

    def __getitem__(self, idx):
        return self.inventory[idx]


kiwi = Kiwi("green", True, 6)
pasta = Pasta("pesto", count=2, max_count=10)
my_inventory = Inventory(10)

my_inventory[0] = kiwi
my_inventory[0] = kiwi
my_inventory[4] = pasta
print(my_inventory[0].count)
my_inventory.get(0, 11)
print(my_inventory[0].count)
print(my_inventory[4].count)
