#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#  estimation.py
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

from collections import MutableMapping, Counter

class EmissionHandler(MutableMapping):
	"""dict-like container that handles emission probability estimation
	
	This class stores emission probabilities of state-emission pairs 
	found in the training data and allows accessing of this information 
	as though from a dict. If a key is given that does not yet have a 
	value then the emission probability is estimated based on the suffix
	of the token and then cached for later use. 
	
	When used in the chunking model no smoothing is necessary since 
	every state/emission pair is seen in the training data (when using 
	universal-dependencies for English, version 1.0). 
	
	attributes:
		data (Counter): stores raw frequencies of state/emission tuples
		Q_counts (Counter): raw frequencies of states (from training)
		S_counts (Counter): raw frequencies of suffixes (from training)
		Q_S_counts (Counter): raw frequencies of state/suffic tuples
		array (numpy.ndarray): normalized array containing emission Ps
			for state/emission pairs found in training data
		found (dict): caches state/emission pairs found outside of
			training data
		token_N (int): number of tokens in training data
		theta (float): weight constant used in smoothing
		converter: converter object used by model
	
	methods:
		add(key): increments frequency of state/emission pair by one
		_P_estimate(*args): estimates P for (state,suffix) or state
		normalize(state_N, emission_N)): create array attribute from
			raw frequencies stored in data attribute
		train(Q_counts, S_counts, Q_S_counts, token_N): set attributes
			and calculate theta attribute
		__getitem__(key): returns emission P if found, otherwise
			caches and returns an estimate"""
	def __init__(self, converter):
		self.data = Counter()
		self.Q_counts = Counter()
		self.S_counts = Counter()
		self.Q_S_counts = Counter()
		self.token_N = 0
		self.array = zeros((1,1))
		self.converter = converter
		self.found = dict()
	
	def __setitem__(self, key, value):
		self.found[key] = value
	
	def __getitem__(self, key):
		"""Access stored emission Ps through (state, emission) keys
		
		If the key does not have a corresponding value an emission P
		is estimated based on token suffix (of max length 5). Smoothing 
		is done using successively shorter suffixes of the same token, 
		as described by (Brants 2000). 
		
		Returns P(emission | state) as float"""
		Q, E, MAX_M = 0, 1, 10
		if key in self.found:
			return self.found[key]
		elif key[E] >= self.array.shape[E]:
			tag = key[Q]
			token = self.converter.decode_tokens(key[E])[0]
			#  Find longest (max M) suffix extant in training data
			suffix = token[-min(len(token), MAX_M):]
			while self.S_counts[suffix] == 0:
				suffix = suffix[1:]
			#  Successive accumulation over length 
			acc = self._P_estimate(tag)
			for i in range(1, len(suffix) + 1):
				acc = (acc * self.theta + 
						self._P_estimate(tag, suffix[-i:])) \
									/ (1 + self.theta)
			self.found[key] = acc
			return self.found[key]
		return self.array[key]
	
	def __delitem__(self, key):
		del self.data[key]
	
	def __iter__(self):
		return iter(self.data)
	
	def __len__(self):
		return len(self.data)
	
	def _P_estimate(self, *args):
		"""Estimates P for state,suffix or state.
		
		P^(state) is frequency of state divided by number of tokens in
		training data. P^(state,suffix) is frequency of state, suffix
		cooccurrance divided by frequency of suffix.
		
		Returns the a priori P^(state) or P^(state | suffix) as float"""
		Q, S = 0, 1
		if len(args) == 1 or len(args[S]) == 0:
			return float(self.Q_counts[args[Q]]) / float(self.token_N)
		return float(self.Q_S_counts[args]) / float(self.S_counts[args[S]])
	
	def train(self, Q_counts, S_counts, Q_S_counts, token_N):
		"""Stores frequency counts and calculates smoothing weight
		
		theta weight is set to the standard deviation of P(q). Raw 
		frequencies of states, suffixes and suffix/state coocurrances
		are needed for smoothing and estimation. This method is run by
		the Model class during training.
		
		arguments:
			Q_counts (Counter): raw frequencies of states
			S_counts (Counter): raw frequencies of suffixes
			Q_S_counts (Counter): cooccurance counts of state/suffix
			token_N (int): the number of tokens in the training data"""
		self.Q_counts = Q_counts
		self.S_counts = S_counts
		self.Q_S_counts = Q_S_counts
		self.token_N = token_N
		#  theta is set to standard deviation of P(tag)
		P_bar = sum(self._P_estimate(tag) for tag in self.Q_counts) \
						/ float(len(self.Q_counts))
		self.theta = sum((self._P_estimate(tag) - P_bar)**2 
									for tag in self.Q_counts) \
							/ float(len(self.Q_counts) - 1)
	
	def normalize(self, state_N, emission_N):
		"""Normalize matrix of found state/emission pairs
		
		This method is run by the Model class during training."""
		array = zeros((state_N, emission_N))
		for key in self.data:
			array[key] = self.data[key]
		self.array = array / matrix(array.sum(axis=1)).getT()
	
	def add(self, key):
		self.data[key] += 1
	
	@staticmethod
	def test():
		pass

class TransitionHandler():
	"""dict-like container that handles transition P estimation
	
	
	
	attributes:
		lambdas (list): list of linear weights used in smoothing
		unigrams (Counter): counts of unigrams
		bigrams (Counter): counts of bigrams
		trigrams (Counter): counts of trigrams
		data (dict): caches previously requested transitions
	
	methods:
		train(unigrams,bigram,trigram,token_N): calculates lambdas
			using context free linear interpolation
		__getitem__(key): calculates transition P using linear smoothing
			unless key is already cached, then simply fetches value."""
	def __init__(self):
		self.lambdas = [0, 0, 0]
		self.unigrams = {}
		self.bigrams = {}
		self.trigrams = {}
		self.data = dict()
	
	def train(self, unigrams, bigrams, trigrams, token_N):
		"""Calculate lambda-weights based on uni- bi- and trigram counts
		
		This method is run by the Model class during training.
		
		Lambda-weights are calculated using context-free interpolation, 
		described in (Brants 2000). Also stores uni-, bi- and 
		trigramcounts."""
		#  find lambdas
		l =  array([0, 0, 0], dtype=float64)
		for t1, t2, t3 in trigrams:
			c3 = (trigrams[t1, t2, t3] - 1.0) / (bigrams[t1, t2] - 1.0) if (bigrams[t1, t2] - 1.0) else  0
			c2 = (bigrams[t2, t3] - 1.0) / (unigrams[t2] - 1.0) if (unigrams[t2] - 1.0) else 0
			c1 = (unigrams[t3] - 1.0) / (token_N - 1.0)
			c = array([c3, c2, c1])
			l[c.argmax()] += trigrams[t1, t2, t3]
		l = l / l.sum()
		#  store attributes
		self.lambdas = [l[2], l[1], l[0]]
		self.unigrams = unigrams
		self.bigrams = bigrams
		self.trigrams = trigrams
		self.token_N = token_N
		for t1, t2, t3 in trigrams:
			self[t3,t1,t2]
	
	def __getitem__(self, key):
		"""Returns transition P of key, estimating it if key is new
		
		key should be of form (k,i,j), with i,j,k being states found
		in that order. Transition P is estimated based on context-
		free linear smoothing using uni-, bi- and trigrams. The value
		is then cached for later use.
		
		Returns transition P of key as float"""
		t3, t1, t2 = key #  key is states i, j, k with k first
		if (t3, t1, t2) not in self.data:
			p1 = self.unigrams.get(t3, 0) / float(self.token_N)
			p2 = self.bigrams.get((t2, t3), 0) / float(self.unigrams[t2]) if self.unigrams.get(t2) else 0
			p3 = self.trigrams.get((t1, t2, t3), 0) / float(self.bigrams[t1, t2]) if self.bigrams.get((t1,t2)) else 0
			self.data[t3,t1,t2] = self.lambdas[0]*p1 + self.lambdas[1]*p2 + self.lambdas[2]*p3
		return self.data[t3,t1,t2]
	
	
	@staticmethod
	def test():
		test_dicts = ({1:55,0:12,2:44,3:5},{(1,2):1,(1,0):5,(0,2):15,(0,1):5,(0,0):2,(0,3):15,(2,3):1,(3,2):1,(2,1):1, (3,1):1, (3,3):1, (2,0):1},{(0,1,2):1,(2,1,2):2,(1,2,3):3,(0,2,0):1,(1,0,3):2,(0,0,3):3,(3,2,0):4,(3,3,1):1,(0,0,0):1,(1,2,1):1,(0,3,1):1},58)
		s = TransitionHandler()
		s.fit(*test_dicts)
		print("TransitionHandler test: ")
		print(s.lambdas)
		print(s[0,1,1], s[3,3,3], s[3,2,1], s[1,2,1])
		print(s.data)
	
	

if __name__ == '__main__':
	pass

