class Util:
    @staticmethod
    def factorization(value):
        prime_numbers = Util.get_prime()
        prime_factors = []
        i = 0
        while value > 1:
            remainder = value % prime_numbers[i]
            if remainder == 0:
                prime_factors.append(prime_numbers[i])
                value /= prime_numbers[i]
                i -= 1
            i += 1
            try:
                prime = prime_numbers[i]
            except IndexError:
                # print('Warning: factorization includes prime numbers greater than 1009')
                return []
        return prime_factors

    @staticmethod
    def primatize(array):
        prime = Util.get_prime()
        group = dict()
        for i, element in enumerate(array):
            group[prime[i]] = element
        return group

    @staticmethod
    def combinations(array, r):  # r is the length of the desired combination arrays
        # TODO this algorithm for finding unique combinations can be improved
        primatized = Util.primatize(array)
        groups = [0] * int(Util.factorial(len(array)) / (Util.factorial(r) * Util.factorial(len(array) - r)))
        value = 2
        for i, _ in enumerate(groups):
            combination = []
            while len(combination) != r or not Util.combination_is_unique(combination) or not Util.test(combination, primatized):
                combination = Util.factorization(value)
                value += 1
            groups[i] = [primatized[j] for j in combination]
        return groups

    @staticmethod
    def test(combination, primification):
        try:
            [primification[j] for j in combination]
        except KeyError:
            return False
        return True

    @staticmethod
    def combination_is_unique(combination):
        for element in combination:
            if combination.count(element) != 1:
                return False
        return True

    @staticmethod
    def simplify_common_factors(numerator, denominator):
        n_factors = Util.factorization(numerator)
        d_factors = Util.factorization(denominator)
        for number in n_factors:
            if number in d_factors:
                n_factors.remove(number)
                d_factors.remove(number)
        numerator, denominator = 1, 1
        for x in n_factors:
            numerator *= x
        for x in d_factors:
            denominator *= x
        return numerator, denominator

    @staticmethod
    def multiply_list_values(list):
        value = list[0] / list[0]
        for item in list:
            value *= item
        return value

    @staticmethod
    def factorial(integer):
        factorial = 1
        for i in range(1, integer+1):
            factorial *= i
        return factorial


########################################################################################################################

    @staticmethod
    def read_prime_numbers():
        f = open('prime_numbers.txt', 'r')
        c = []
        for a in f:
            for b in a.split():
                c.append(int(b))
        return c
        f.close()

    @staticmethod
    def get_prime():
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
                211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331,
                337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457,
                461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
                601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
                739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877,
                881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009]


if __name__ == '__main__':
    primatization = Util.primatize(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o'])
    comb = Util.combinations(primatization, 3)
    print('length', len(comb))