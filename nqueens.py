import random
import numpy as np
import sys
from functools import reduce
from operator import mul
from terminaltables import SingleTable


class Population:
	def __init__(self, size, amount):
		self.size   = size
		self.amount = amount
		self.best_fit = ()
		self.best_fit_fitness = -1

	def randomise(self):
		return tuple(random.sample(range(0, self.size), self.size))

	def setup(self):
		return [self.randomise() for i in range(0, self.amount)]

	def factorial(self, n):
		if n < 2:
			return 1
		return [reduce(mul, [i for i in range(1, n + 1)])][0]

	def combination(self, n):
		if n < 2:
			return 0
		return self.factorial(n) / (self.factorial(2) * self.factorial(n - 2))

	def generate(self, layout):
		board = [[0 for y in range(0, self.size)] for x in range(0, self.size)]
		for c, v in enumerate(list(layout)):
			board[c][v] = 1
		return board

	def evaluate(self, layout):
		board = self.generate(layout)
		fitness = 0

		for c in range(0, len(board[0])):
			fitness += self.combination(sum([row[c] for row in board]))

		for d in range(0, len(board)):
			right_diagonal, left_diagonal = 0, 0
			hrx, hry, vrx, vry = 0, d, d + 1, 0
			hlx, hly, vlx, vly = self.size - 1, d, self.size - 2 - d, 0

			while hrx < len(board) and hry < len(board):
				right_diagonal += board[hrx][hry]
				hrx += 1
				hry += 1

			while hlx >= 0 and hly < len(board):
				left_diagonal += board[hlx][hly]
				hlx -= 1
				hly += 1

			fitness += self.combination(right_diagonal)
			fitness += self.combination(left_diagonal)

			right_diagonal, left_diagonal = 0, 0

			while vrx < len(board) and vry < len(board):
				right_diagonal += board[vrx][vry]
				vrx += 1
				vry += 1

			while vlx >= 0 and vly < len(board):
				left_diagonal += board[vlx][vly]
				vlx -= 1
				vly += 1

			fitness += self.combination(right_diagonal)
			fitness += self.combination(left_diagonal)

		return (self.combination(self.size) - fitness)

	def crossover(self, parent1, parent2):
		slice = np.random.randint(0, self.size - 1)
		return parent1[:slice] + parent2[slice:]

	def mutate(self, parent):
		mutant = random.randint(0, self.size - 1)
		return parent[:mutant] + tuple([random.randint(0, self.size - 1)]) + parent[mutant + 1:]
	

	def evolve(self, old_population, mutation_rate):
		fitness = {}
		total_fitness = 0
		new_population = set()

		for layout in old_population:
			layout_fitness = self.evaluate(layout)

			if layout_fitness > self.best_fit_fitness:
				self.best_fit, self.best_fit_fitness = tuple(layout), layout_fitness

			total_fitness += layout_fitness

			fitness[layout] = layout_fitness

		previous = 0.0

		for l, lf in fitness.items():
			previous += (lf / total_fitness)
			fitness[l] = previous * 100

		layouts = []

		for layout in fitness.keys():
			for i in range(0, int(fitness[layout])):
				layouts.append(layout)

		for j in range(0, len(old_population) - int(self.amount * 0.01)):
			p1, p2 = np.random.randint(len(layouts)), np.random.randint(len(layouts))

			parent1, parent2 = layouts[p1], layouts[p2]

			child = self.crossover(parent1, parent2)

			mutation_chance = np.random.random_sample()

			if mutation_chance < mutation_rate:
				child = self.mutate(child)

			new_population.add(child)

		for k in range(0, int(self.amount * 0.01)):
			new_population.add(self.randomise())

		return new_population

	def goal_is_reached(self, population):
		for layout in population:
			if self.evaluate(layout) == self.combination(self.size):
				self.best_fit = tuple(layout)
				return True

		return False

	def show(self, layout):
		board = self.generate(list(layout))
		table_data = []

		for x in range(0, len(board)):
			row = []
			for y in range(0, len(board[x])):
				if board[x][y] == 1:
					row.append('♕ ')
				else:
					row.append(' ')

			table_data.append(row)


		table = SingleTable(table_data)
		table.inner_row_border = True
		print(table.table)


size, amount, mutation_rate = sys.argv[1], sys.argv[2], sys.argv[3]
p = Population(int(size), int(amount))

population = p.setup()
gen = 0

while True:
	if gen > 10000:
		print("Solution not found :(")
		break

	if p.goal_is_reached(population):
		print("Solution found at GEN " + str(gen) + ":")
		p.show(p.best_fit)
		break

	population = p.evolve(population, float(mutation_rate))
	gen += 1
