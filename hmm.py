#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#  hmm.py
#  
#  Copyright 2015 Peter Persson <peter.johan.persson@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

from numpy import *
from time import time
from operator import itemgetter

from converter import Converter
from estimation import TransitionHandler, EmissionHandler
from model import Model
from collections import Counter, deque
from textutils import ConllParser


CHUNK = 0
POS = 1

class HMM:
	"""Mostly superfluous class that only contains a single method.
	See the documentation for that method."""
	#Jungyeul Park, Mouna Chebbah, Siwar Jendoubi, Arnaud Martin. Second-Order Belief Hidden Markov Models. Belief 2014, Sep 2014, Oxford, United Kingdom. pp.284 - 293, 2014, <10.1007/978-3-319-11191-9_31>.<hal-01108238>
	
	def viterbi(self, token_list, model):
		"""Find optimal hidden path for token_list using beam search
		
		Uses a second order HMM model and decodes the most likely state
		path using a beam search. All symbols are internally handled as
		integers and need to be decoded with the converter object of the
		model argument. Method and maths used is from Thorsten Brants
		(2000) and supplemented with details from Park et al. (2014).
		
		arguments: 
			token_list (list): a list of integers representing 
				emissions (tokens when POS tagging)
			model (Model): the model supplies transition and emission
				probabilities as well as symbol language
		
		Returns a deque object with the most likely state path"""
		observations = model.converter.convert_tokens(*token_list)
		#  E for emission, Q for state, T for length of input list
		E, Q, T, N_STATES = 0, 1, len(observations), model.get_state_N()
		BEAM_C = 1/1000 #  beam search threshold constant
		a, b = model.transitions, model.emissions
		bt = dict()
		#  viterbi if sentence is exceptionally short
		if T == 1:
			v0 = array([a[k, model.S0_Q, model.S1_Q] \
						* b[k, observations[0]] \
						* a[model.END_Q, model.S1_Q, k]
								for k in range(N_STATES)])
			return [v0.argmax()]
		if T == 2:
			v0 = array([a[k, model.S0_Q, model.S1_Q] \
						* b[k, observations[0]] 
								for k in range(N_STATES)])
			k_list = []
			for k in range(N_STATES):
				j_list = array([a[k, model.S1_Q, j] * v0[j] \
								* b[k, observations[1]] \
								* a[model.END_Q, j, k]
									for j in range(N_STATES)])
				k_list.append((j_list.max(), j_list.argmax()))
			Q1, q_tuple = max(enumerate(k_list), key=itemgetter(1))
			Q0 = q_tuple[1]
			return [Q0,Q1]
		#  initialize first row w. beam
		v0 = array([a[k, model.S0_Q, model.S1_Q] \
					* b[k, observations[0]] for k in range(N_STATES)])
		threshold = max(v0) * BEAM_C
		beam_j = [j for j in range(N_STATES) if v0[j] >= threshold]
		#  initialize second row w. beam
		vt = zeros((N_STATES, N_STATES))
		for k in range(N_STATES):
			for j in beam_j:
				vt[k, j] = a[k, model.S1_Q, j] * v0[j] \
							* b[k, observations[1]]
		threshold = max([max(vt[k]) for k in range(N_STATES)]) * BEAM_C
		beam_k = [k for k in range(N_STATES) \
						if not max(vt[k]) < threshold]
			
		#  recursive step
		for t in range(2, T + 1, 1):
			beam_i = beam_j
			beam_j = beam_k
			v0 = vt
			vt = zeros((N_STATES, N_STATES))
			#  i, j, k represent states in a trigram under consideration
			for k in range(N_STATES):
				for j in beam_j:
					i_list = [] #  list of viterbi values per i, j, k
					for i in beam_i:
						if t < T : i_list.append((a[k,i,j] \
											* v0[j, i] \
											* b[k, observations[t]], i))
						else: i_list.append((a[k,i,j] * v0[j, i] \
											 * b[k, model.END_E], i))
					best_P, best_i = max(i_list)
					vt[k, j] = best_P
					#  set backtracing values
					if k == 0: bt[t - 1, j] = zeros((N_STATES))
					bt[t - 1, j][k] = best_i
				if t == T:
					bt[t, k] = vt[k].argmax()
			threshold = max([max(vt[k]) for k in range(N_STATES)]) \
						  * BEAM_C
			beam_k = [k for k in range(N_STATES) \
						if not max(vt[k]) < threshold]
		#  initialize backtracing
		path = deque()
		path.appendleft(model.END_Q)
		j = bt[T, model.END_Q]
		#  recursive step
		for t in range(T - 1, 0, -1):
			path.appendleft(j)
			j = int(bt[t, j][path[1]])
		# terminate and remove 'END' node at the end of path
		path.appendleft(j)
		path.pop() 
		
		return path
	
	@staticmethod
	def test():
		test_sentence = "Det här är en testmening ."
		testfile = "pos_model.pickle"
		hmm = HMM()
		model = Model()
		model.load_from(testfile)
		t0 = time()
		#print(*hmm.viterbi(test_sentence.split(), model))
		print(model.converter.decode_tags(*hmm.viterbi(test_sentence.split(), model)))
		t = t0 - time()
		print(t)
	
	@staticmethod
	def test_POS_UD(filename, filesize):
		testmodel = "pos_model_UD_en.pickle"
		hmm = HMM()
		parser = ConllParser()
		model = Model()
		model.load_from(testmodel)
		correct_n, total_n, line_n, progress = 0, 0, 0, 0
		with open(filename, 'r') as inf:
			sentence, tags = [], []
			line = inf.readline()
			while line:
				if line != "\n":
					token, tag, a = parser.parse_line_POS(line)
					sentence.append(token)
					tags.append(model.converter.convert_state(tag))
				else:
					if len(sentence) > 0:
						guesses = hmm.viterbi(sentence, model)
						for i in range(len(sentence)):
							if tags[i] == guesses[i]: correct_n += 1
							total_n += 1
						sentence, tags = [], []
						progress_value = int(line_n * 100 / filesize) % 10
						if progress_value > progress:
							progress = progress_value
							print( str(progress*10)+"%")
							print("Accuracy:", correct_n / float(total_n))
				line = inf.readline()
				line_n += 1
			print("Final accuracy:", correct_n / float(total_n))
		
	@staticmethod
	def test_CHUNK_UD(filename, filesize):
		testmodel = "chunk_model_UD_en.pickle"
		hmm = HMM()
		parser = ConllParser()
		model = Model()
		model.load_from(testmodel)
		correct_n, total_n, line_n, progress = 0, 0, 0, 0
		with open(filename, 'r') as inf:
			sentence, tags = [], []
			line = inf.readline()
			while line:
				if line != "\n":
					token, tag = parser.parse_line_CHUNK(line)
					sentence.append(token)
					tags.append(model.converter.convert_state(tag))
				else:
					if len(sentence) > 0:
						guesses = hmm.viterbi(sentence, model)
						for i in range(len(sentence)):
							if tags[i] == guesses[i]: correct_n += 1
							total_n += 1
						sentence, tags = [], []
						progress_value =  int(line_n * 100 / filesize / 10)
						if progress_value > progress:
							progress = progress_value
							print( str(progress*10)+"%")
							print("Accuracy:", correct_n / float(total_n))
				line_n += 1
				line = inf.readline()
		print("Final accuracy:", correct_n / float(total_n))
					
		

if __name__ == '__main__':
	#Converter.test()
	#TransitionHandler.test()
	#EmissionHandler.test()
	#HMM.test_POS_UD("/home/peterpersson/lin503/projekt/universal_treebanks_v1.0/en-universal-test.conll", 59100)
	HMM.test_CHUNK_UD("/home/peterpersson/lin503/projekt/universal_treebanks_v1.0/en-universal-test.conll", 59100)
	#HMM.test()
