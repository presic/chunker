#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#  syntaxtranslator.py
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
 
from textutils import ConllParser
from collections import deque

class Translator:
	"""Converts universal stanford dependencies into IOB2 tags
	
	Reads through a file using parse_tree method from textutils, then 
	converts the tree into a list representation of phrases which is
	then finally turned into IOB2 tags. Discontinuous phrases are 
	treated as separate chunks and the set of chunk types used is the 
	following:
	
	NP complete non-recursive noun phrases
	VC verb clusters incl. copulas and modal verbs
	INFP verb clusters (with particles) that are non-finite arguments 
	AP adjective phrases not inside of NPs
	ADJP adjective phrases not inside of NPs or APs
	PP adpositions only, dominated NPs are their own chunks
	CONJ conjunction words
	SUBJ subjunction words
	
	"""
	def __init__(self):
		self.conll = ConllParser()
	
	def annotate_file(self, infile, outfile, column=5):
		"""Write a copy of infile at outfile enriched with IOB-2 tags
		
		infile must be a .conll file with standford dependencies. 
		column is where in the file the new annotation will go. 
		"""
		with open(infile, 'r') as inf, open(outfile, 'w') as outf:
			source_lines = [1]
			while source_lines[-1]: 
				#  bool(source_lines[-1]) is false when last line is EOF
				tree, source_lines = self.conll.parse_tree(inf, keep_lines=True)
				if not tree: continue
				chunk_tags = self.translate_tree(tree)
				for i, line in enumerate(source_lines):
					if line == '\n': outf.write(line)
					elif line == '': continue
					else:
						data = line.split("\t")
						data[column] = chunk_tags[i]
						outf.write("\t".join(data))
	
	def translate_tree(self, tree):
		"""Finds chunks from tree, returning list of IOB2 tags
		
		tree must be of the format used in textutils module. This
		function is currently not safe for incorrect input. 
		
		The algorithm does a breadth-first search over the tree creating
		new phrases where a dependency type that heads a phrase is found
		(see dependency classes below) and adds the current node to its
		parent's phrase otherwise. Special consideration is given to 
		ROOT depenceny as well as some others. Lastly phrases are turned
		into IOB2 tags treating discontinuous phrases as separate 
		chunks."""
		
		#  define dependency classes
		debug_KNOWN_DEPS = {'nsubj', 'nsubjpass', 'iobj', 'dobj', 
			'nmod', 'rel', 'expl', 'adpobj', 'attr', 'poss', 'appos',
			'parataxis', 'csubj', 'csubjpass', 'advcl', 'rcmod', 
			'ccomp', 'adpcomp','infmod', 'xcomp','adpmod','acomp', 
			'amod','advmod', 'partmod','cc','mark','dep', 'det','aux',
			'auxpass','compmod','mwe','neg','num','p','prt','vmod', 
			'ROOT', 'adp', 'cop'}
		NP_HEADS = {'nsubj', 'nsubjpass', 'iobj', 'dobj', 'nmod', 'rel',
					'expl', 'attr', 'appos', 'adpobj'}
		VC_HEADS = {'parataxis', 'csubj', 'csubjpass', 'advcl', 'rcmod',
					'ccomp', 'adpcomp', 'vmod', 'cop', 'partmod'}
		INFP_HEADS = {'infmod', 'xcomp'}
		PP_HEADS = {'adpmod'} 
		AP_HEADS = {'acomp', 'amod'} 
		ADVP_HEADS = {'advmod'}
		CONJ_HEADS = {'cc'}
		SUBJ_HEADS = {'mark'} #sensitive to parent dep type
		O = {'p'}
		
		#  phrases are represented as ['type', index1, index2, ...]
		phrase_list = [['O']]
		children = deque([tree[0]])
		#  loop over tree breadth-first 
		while children:
			node = tree[children.popleft()]
			parent = tree[node.parent]
			try:
				parent_phrase_type = phrase_list[parent.phrase][0]
			except AttributeError:
				parent_phrase_type = 'ROOT'
			except TypeError:
				print(parent, parent.phrase, parent.pos_tag)
				print(node)
				raise 
			#  NPs always make their own phrases
			if node.dep_type in NP_HEADS:
				node.phrase = len(phrase_list)
				phrase_list.append(["NP", node.index])
			#  VCs
			elif node.dep_type in VC_HEADS:
				node.phrase = len(phrase_list)
				phrase_list.append(["VC", node.index])
			#  INFPs
			elif node.dep_type in INFP_HEADS:
				node.phrase = len(phrase_list)
				phrase_list.append(["INFP", node.index])
			#  PPs
			elif node.dep_type in PP_HEADS:
				node.phrase = len(phrase_list)
				phrase_list.append(["PP", node.index])
			#  CONJs 
			elif node.dep_type in CONJ_HEADS:
				node.phrase = len(phrase_list)
				phrase_list.append(["CONJ", node.index])
			#  ADVPs
			elif node.dep_type in ADVP_HEADS:
				if parent_phrase_type not in {'NP', 'AP', 'PP', 'ADVP'}:
					node.phrase = len(phrase_list)
					phrase_list.append(["ADVP", node.index])
				else:
					node.phrase = parent.phrase
					phrase_list[node.phrase].append(node.index)
			#  APs
			elif node.dep_type in AP_HEADS:
				if parent_phrase_type not in {'NP', 'AP', 'PP'}:
					node.phrase = len(phrase_list)
					phrase_list.append(["AP", node.index])
				else:
					node.phrase = parent.phrase
					phrase_list[node.phrase].append(node.index)
			#  O 
			elif node.dep_type in O: 
				node.phrase = 0
				phrase_list[0].append(node.index)
			#  conj is added to parent's phrase with parent's dep type
			elif node.dep_type == 'conj': 
				node.dep_type = parent.dep_type
				node.phrase = parent.phrase
				phrase_list[node.phrase].append(node.index)
			#  mark defaults to subj unless marking clausal adverbial
			elif node.dep_type == 'mark':
				node.phrase = len(phrase_list)
				if parent.dep_type == 'advcl':
					phrase_list.append(["ADVP", node.index])
				else:
					phrase_list.append(["SUBJ", node.index])
			#  ROOT
			elif node.dep_type == 'ROOT':
				if node.pos_tag == 'VERB':
					node.phrase = len(phrase_list)
					phrase_list.append(["VC", node.index])
				elif node.pos_tag == 'NOUN':
					node.phrase = len(phrase_list)
					phrase_list.append(["NP", node.index])
				elif node.pos_tag == 'ADP':
					node.phrase = len(phrase_list)
					phrase_list.append(["PP", node.index])
				elif node.pos_tag == 'PRON':
					node.phrase = len(phrase_list)
					phrase_list.append(["NP", node.index])
				elif node.pos_tag == 'ADJ':
					node.phrase = len(phrase_list)
					phrase_list.append(["AP", node.index])
				elif node.pos_tag == 'ADV':
					node.phrase = len(phrase_list)
					phrase_list.append(["ADVP", node.index])
				elif node.pos_tag == 'NUM':
					node.phrase = len(phrase_list)
					phrase_list.append(["ADVP", node.index])
				elif node.pos_tag == 'PRT':
					node.phrase = len(phrase_list)
					phrase_list.append(["VC", node.index])
				elif node.pos_tag == 'CONJ':
					node.phrase = len(phrase_list)
					phrase_list.append(["CONJ", node.index])
				#  Some of [NP], first half [of NP]
				elif node.pos_tag == 'DET': 
					node.phrase = len(phrase_list)
					phrase_list.append(["NP", node.index])
				#  mostly $ as head of '$ 100 million', rest is garbage
				elif node.pos_tag == '.': 
					node.phrase = len(phrase_list)
					phrase_list.append(["NP", node.index])
				#  mostly noise, some are 'oh yeah!' 'yeah!' 'damn!'
				elif node.pos_tag == 'X': 
					node.phrase = len(phrase_list)
					phrase_list.append(["ADVP", node.index])
				else:
					print('ROOT:',node.pos_tag)
					raise TypeError()	
			#  default: join parent's phrase
			else:
				node.phrase = parent.phrase
				phrase_list[node.phrase].append(node.index)
			"""finding incorrect annotation 
			if node.dep_type not in debug_KNOWN_DEPS: 
				print(node.dep_type)
			"""
			# update loop
			children.extend(node.children)
			
		#  convert phrases to tags in a list
		chunk_tags = [0]* len(tree)
		for phrase in phrase_list:
			if phrase[0] == 'O':
				for i in range(1, len(phrase)):
					chunk_tags[phrase[i]] = 'O'
			else:
				node_indexes = sorted(phrase[1:])
				for i, index in enumerate(node_indexes):
					if i == 0: chunk_tags[index] = 'B-'+ phrase[0]
					#  disjunct phrases are treated as separate entities
					elif node_indexes[i-1] == index - 1:
						chunk_tags[index] = 'I-'+ phrase[0]
					else: 
						#  punctuation does not split phrases
						if tree[index - 1].phrase == 0 \
								and node_indexes[i-1] == index - 2:
							chunk_tags[index - 1] = 'I-'+ phrase[0]
							chunk_tags[index] = 'I-'+ phrase[0]
						else:
							chunk_tags[index] = 'B-'+ phrase[0]
		return chunk_tags[1:]
	@staticmethod
	def test(infile, outfile):
		t = Translator()
		t.annotate_file(infile, outfile)
		

if __name__ == '__main__':
	short_files = "/home/peterpersson/lin503/projekt/universal_treebanks_v2.0/std/test.conll", "/home/peterpersson/lin503/projekt/universal_treebanks_v2.0/std/test-output.conll"
	#train_files = "/home/corpora/universal_treebanks_v2.0/std/de/de-universal-train.conll","/home/peterpersson/lin503/projekt/universal_treebanks_v2.0/std/de-universal-train.conll"
	#dev_files = "/home/corpora/universal_treebanks_v2.0/std/de/de-universal-dev.conll","/home/peterpersson/lin503/projekt/universal_treebanks_v2.0/std/de-universal-dev.conll"
	#test_files = "/home/corpora/universal_treebanks_v2.0/std/de/de-universal-test.conll","/home/peterpersson/lin503/projekt/universal_treebanks_v2.0/std/de-universal-test.conll"
	train_files = "/home/corpora/universal_treebanks_v1.0/en/en-univiersal-train.conll","/home/peterpersson/lin503/projekt/universal_treebanks_v1.0/en-universal-train.conll"
	dev_files = "/home/corpora/universal_treebanks_v1.0/en/en-univiersal-dev.conll","/home/peterpersson/lin503/projekt/universal_treebanks_v1.0/en-universal-dev.conll"
	test_files = "/home/corpora/universal_treebanks_v1.0/en/en-univiersal-test.conll","/home/peterpersson/lin503/projekt/universal_treebanks_v1.0/en-universal-test.conll"
	
	Translator.test(*train_files)
	Translator.test(*dev_files)
	Translator.test(*test_files)

