#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#  model.py
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
import pickle
from converter import Converter
from estimation import TransitionHandler, EmissionHandler
from textutils import ConllParser
from collections import Counter

CHUNK = 0
POS = 1

class Model():
	"""Handles emission and transition probabilities and training
	
	Model reads data from a corpus with train() function. Before train()
	model doesn't do anything or store any data. All names and 
	probabilities used by the HMM class is stored in a model object, 
	which is passed to the HMM class' viterbi(). 
	
	attributes: 
		converter: Converter object storing int names of state/emission
		transitions: TransitionHandler used to access transition Ps
		emissions: EmissionHandler used to access emission Ps
		conll: ConllParser object for reading from conll files
		
		S0_Q (int): int name of first start state symbol
		S1_Q (int): int name of second start state symbol
		END_Q (int): int name of end state symbol
		S0_E (int): int name of first start emission symbol
		S1_E (int): int name of second start emission symbol
		END_E (int): int name of end emission symbol
		
	methods:
		get_state_N(): returns number of states in model
		get_emission_N(): returns number of emissions in model
		train(filename, mode): train model from conll file at filename
		save_at(filename): pickle dump to filename
		load_from(filename): unpickle from filename
	"""
	def get_state_N(self):
		return self.converter.get_state_N()
		
	def get_emission_N(self):
		return self.converter.get_emission_N()
	
	def train(self, filename, mode=POS):
		"""Learns probabilities and symbol names from conll file
		
		Reads data from conll file at filename and trains a Converter
		object, a TransitionHandler object and an Emission handler 
		object from the data. Also defines start and end symbols.
		
		arguments:
			filename (string): the conll file to learn from
			mode (int): should be CHUNK or POS depending on model type
				to be trained
		"""
		#  E for emission, Q for state, S for suffix
		E, Q, S, BLANK = 0, 1, 2, "\n"
		#  clear/init model
		self.converter = Converter()
		self.transitions = TransitionHandler()
		self.emissions = EmissionHandler(self.converter)
		self.conll = ConllParser()
		#  define metrics
		token_N = 0
		state_N = 0
		trigrams = Counter()
		bigrams = Counter()
		unigrams = Counter()
		S_counts = Counter()
		Q_counts = Counter()
		Q_S_counts = Counter()
		#  defining special symbols
		self.S0_Q = self.converter.convert_state('S0')
		self.S1_Q = self.converter.convert_state('S1')
		self.S0_E = self.converter.convert_emission('S0')
		self.S1_E = self.converter.convert_emission('S1')
		self.END_Q = self.converter.convert_state('END')
		self.END_E = self.converter.convert_emission('END')
		#  add them to uni- and bigrams and set emissions
		unigrams[self.S0_Q] += 1; unigrams[self.S1_Q] += 1
		bigrams[self.S0_Q, self.S1_Q] += 1
		self.emissions.add((self.END_Q, self.END_E))
		self.emissions.add((self.S0_Q, self.S0_E))
		self.emissions.add((self.S1_Q, self.S1_E))
		#  read data from file
		with open(filename, 'r') as inf:
			sentence = []
			line = inf.readline()
			while line:
				if line != BLANK:
					if mode == POS: 
						entry = self.conll.parse_line_POS(line)
					else: entry = self.conll.parse_line_CHUNK(line)
					#  convert/learn int names
					q = self.converter.convert_state(entry[Q]) 
					e = self.converter.convert_emission(entry[E])
					#  counts for emission
					self.emissions.add((q,e))
					if mode == POS:
						Q_counts[q] += 1
						for s in entry[S]:
							S_counts[s] += 1
							Q_S_counts[q, s] += 1
					#  uni-, bi-, and trigram counts for transition
					if len(sentence) == 0:
						trigrams[self.S0_Q, self.S1_Q, q] += 1
						bigrams[self.S1_Q, q] += 1
						unigrams[q] += 1
					elif len(sentence) == 1:
						trigrams[self.S1_Q, sentence[-1][Q], q] += 1
						bigrams[sentence[-1][Q], q] += 1
						unigrams[q] += 1
					else:
						trigrams[sentence[-2][Q], sentence[-1][Q], q] += 1
						bigrams[sentence[-1][Q], q] += 1
						unigrams[q] += 1
					#  loop update
					sentence.append((q,e))
					token_N += 1
				else:
					#  blank line means end of sentence
					if len(sentence)>= 2: 
						trigrams[sentence[-2][Q], sentence[-1][Q], self.END_Q] += 1
					if len(sentence)>= 1: 
						bigrams[sentence[-1][Q], self.END_Q] += 1
					if len(sentence) > 0: 
						unigrams[self.END_Q] += 1
						Q_counts[self.END_Q] += 1
						Q_counts[self.S0_Q] += 1
						Q_counts[self.S1_Q] += 1
					#  loop update
					sentence = []
				line = inf.readline()
		#  normalize found emissions and train estimations
		state_N = self.converter.get_state_N()
		emission_N = self.converter.get_emission_N()
		self.emissions.normalize(state_N, emission_N)
		if mode == POS: 
			self.emissions.train(Q_counts, S_counts, Q_S_counts, token_N)
		self.transitions.train(unigrams, bigrams, trigrams, token_N)
	
	def save_at(self, filename):
		with open(filename, 'wb') as outf:
			pickle.dump(self.transitions, outf)
			pickle.dump(self.emissions, outf)
	
	def load_from(self, filename):
		with open(filename, 'rb') as inf:
			self.transitions = pickle.load(inf)
			self.emissions = pickle.load(inf)
		self.converter = self.emissions.converter
		#  relearn special symbols 
		self.S0_Q, self.S1_Q = self.converter.convert_tags("S0", "S1")
		self.S0_E, self.S1_E,  = self.converter.convert_tokens("S0", "S1")
		self.END_E, self.END_Q = self.converter.convert_both(("END","END"))[0]
	
	@staticmethod
	def test_POS():
		#testfile = "/home/peterpersson/lin503/projekt/universal_treebanks_v2.0/std/de-universal-train.conll"
		testfile = "/home/peterpersson/lin503/projekt/universal_treebanks_v1.0/en-universal-train.conll"
		m = Model()
		m.train(testfile)
		print(m.get_state_N(), m.get_emission_N())
		m.save_at("pos_model_UD_en.pickle")
		m.load_from("pos_model_UD_en.pickle")
		print(m.get_state_N(), m.get_emission_N())
		
	@staticmethod
	def test_CHUNK():
		#testfile = "/home/peterpersson/lin503/projekt/universal_treebanks_v2.0/std/de-universal-train.conll"
		testfile = "/home/peterpersson/lin503/projekt/universal_treebanks_v1.0/en-universal-train.conll"
		m = Model()
		m.train(testfile, mode=CHUNK)
		print(m.get_state_N(), m.get_emission_N())
		m.save_at("chunk_model_UD_en.pickle")
		m.load_from("chunk_model_UD_en.pickle")
		print(m.get_state_N(), m.get_emission_N())
"""
if __name__ == '__main__':
	Model.test_POS()
	Model.test_CHUNK()
"""
