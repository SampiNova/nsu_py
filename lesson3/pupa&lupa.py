class Worker:
    def __init__(self, money=0):
        self._money = money

    @property
    def money(self):
        return self._money

    def take_salary(self, summ):
        self._money += summ

    @staticmethod
    def read_mat(filename1, filename2):
        file1 = open(filename1, "r")
        file2 = open(filename2, "r")

        mat1 = list(map(lambda x: list(map(int, x[:-1].split(' '))), file1.readlines()))
        mat2 = list(map(lambda x: list(map(int, x[:-1].split(' '))), file2.readlines()))

        file1.close()
        file2.close()
        return mat1, mat2


class Pupa(Worker):
    def __int__(self, money=0):
        super().__init__(money=money)
        pass

    def do_work(self, filename1, filename2):
        mat1, mat2 = super().read_mat(filename1, filename2)
        for r1, r2 in zip(mat1, mat2):
            for c1, c2 in zip(r1, r2):
                print(c1 + c2, end=' ')
            print()


class Lupa(Worker):
    def __int__(self, money=0):
        super().__init__(money=money)

    def do_work(self, filename1, filename2):
        mat1, mat2 = super().read_mat(filename1, filename2)
        for r1, r2 in zip(mat1, mat2):
            for c1, c2 in zip(r1, r2):
                print(c1 - c2, end=' ')
            print()


class Accountant:
    def __init__(self, salary):
        self._salary = salary

    @property
    def salary(self):
        return self._salary

    def give_salary(self, worker):
        worker.take_salary(self._salary)


pupa = Pupa(20)
lupa = Lupa(10)
acc = Accountant(5)

acc.give_salary(pupa)
acc.give_salary(lupa)

print(f"pupa: {pupa.money}\nlupa: {lupa.money}")

pupa.do_work("file1.txt", "file2.txt")
lupa.do_work("file1.txt", "file2.txt")
