#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#  textutils.py
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

from collections import deque

class Node:
	"""Node of a stanford dependency tree
	
	Standford trees are represented as lists of nodes where the first 
	element is the index of the head of the tree. 
	
	attributes:
		self.parent (int): index of parent node
		self.dep_type (string): this node's dependency type
		self.children (list): list of children's indexes
		self.pos_tag (string): this node's POS tag
		self.phrase (list): list representation of phrase (or None)
		self.index (int): this node's index"""
	def __init__(self, parent, dep_type, pos_tag, index):
		self.parent = parent
		self.dep_type = dep_type
		self.index = index
		self.children = []
		self.pos_tag = pos_tag
		self.phrase = None
	
	def __str__(self):
		output = str(self.index) + " " + self.dep_type + " " + self.pos_tag
		if self.phrase: output += " " + str(self.phrase)
		output += " " + str(self.parent) + "\n" + str(self.children)
		return output
	
	def add_child(self, child):
		self.children.append(child.index)

class Tokenizer:
	
	#  placeholder, originally had higher ambitions 
	
	def process(self, sentence):
		return sentence.split()

class ConllParser:
	"""Contains all .conll reading functions."""
	TOKEN, TAG, CHUNK, PARENT, DEP = 1, 3, 5, 6, 7
	def parse_line_POS(self, line):
		"""Used by model to train for POS tagging.
		
		Returns (token, tag)."""
		data = line.split('\t')
		suffixes = self.find_suffixes(data[ConllParser.TOKEN])
		return (data[ConllParser.TOKEN], data[ConllParser.TAG], suffixes)
	
	def parse_line_CHUNK(self, line):
		"""Used by model to train for chunk tagging.
		
		Returns (tag, chunk)."""
		data = line.split('\t')
		return (data[ConllParser.TAG], data[ConllParser.CHUNK])
	
	def parse_line_EVAL(self, line):
		"""Used by evaluation function.
		
		Returns (token, chunk)."""
		data = line.split('\t')
		return (data[ConllParser.TOKEN], data[ConllParser.CHUNK])
	
	def parse_line_TAG(self, line):
		"""Used by tagging function.
		
		Returns token."""
		data = line.split('\t')
		return data[ConllParser.TOKEN]
		
	
	def find_suffixes(self, token):
		"""Returns all suffix strings up to length 10."""
		MAX_M = 10
		return [token[-i:] for i in range(min(len(token), MAX_M))]
	
	
	def parse_tree(self, fileobject, keep_lines=False):
		"""Reads one sentence from fileobject and generates syntax tree
		
		Standford trees are represented as lists of instances of the 
		Node class where the first element is the index of the head of 
		the tree. The function tests for a connected tree (starting at 
		root) before returning.
		
		Returns a list representation of stanford tree. 
		"""
		STOP = ["","\n"]
		source_lines = []
		line = fileobject.readline()
		source_lines.append(line)
		tree = ["ROOT"]
		while line not in STOP:
			data = line.split('\t')
			tree.append(Node(int(data[ConllParser.PARENT]), 
							 data[ConllParser.DEP], 
							 data[ConllParser.TAG], 
							 len(tree)))
			line = fileobject.readline()
			source_lines.append(line)
		for i in range(1, len(tree)):
			node = tree[i]
			if node.parent != 0:
				tree[node.parent].add_child(node)
			else: tree[0] = i
		#  check for connected tree
		try:
			counter = 1
			children = deque([tree[0]])
			while children:
				node = tree[children.popleft()]
				children.extend(node.children)
				counter += 1
			if counter == len(source_lines): corrupt = False
			else: corrupt = True 
		except:
			corrupt = True
		if keep_lines and corrupt: return [], source_lines
		elif keep_lines: return tree, source_lines
		elif corrupt: return []
		return tree
	
	@staticmethod
	def test_tree(filename):
		conll = ConllParser()
		with open(filename, 'r') as inf:
			tree = conll.parse_tree(inf)
		children = deque([tree[0]])
		while children:
			node = tree[children.popleft()]
			print(node)
			children.extend(node.children)
			
			


if __name__ == '__main__':
	ConllParser.test_tree("/home/corpora/universal_treebanks_v2.0/std/de/de-universal-dev.conll")

