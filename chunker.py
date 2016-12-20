#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#  chunker.py
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

import pickle
import os
from hmm import HMM 
from model import Model
from textutils import ConllParser
from argparse import ArgumentParser


#These constants are used for ease-of-reading purposes
CHUNK = 0
POS = 1

class Chunker:
	"""High-level control class for the entire program
	
	Chunker class handles commandline interface and keepds track of 
	models for PoS tagging and Chunk tagging. Also handles input and 
	output to files, outside of loading and saving models which is done 
	by the model class.
	
	attributes:
		pos_model (Model): the model object trained for PoS tagging 
		chunk_model (Model): the model object trained for chunking 
		hmm (HMM): the HMM object used together with either model to 
			perform tagging operations
	methods:
		load_model(filename[, mode]): filename is the filepath for the 
			file containing the model to be loaded, mode is an int 
			specifying whether the model to be loaded should be used 
			for chunking or PoS-tagging
		save_model(filename[, mode]): filename is the filepath the model 
			will be saved at, mode is an int specifying which model to 
			be saved, chunktagging model or postagging model
		tag(tokens, mode): finds pos tags for tokens, if mode is set 
			to chunk find chunk tags to the generate pos tags
		tag_file(infile[, outfile, mode]): generate annotated version of
			file at infile, output either to terminal or outfile
	
	"""
	def __init__(self):
		self.pos_model = None
		self.chunk_model = None
		self.hmm = HMM()
	
	def load_model(self, filename, mode=CHUNK):
		"""Makes new model object by loading from filepath"""
		if mode == CHUNK: 
			self.chunk_model = Model()
			self.chunk_model.load_from(filename)
		if mode == POS: 
			self.pos_model = Model()
			self.pos_model.load_from(filename)
	
	def save_model(self, filename, mode=CHUNK):
		"""Tells model object to save at filepath"""
		try:
			if mode == CHUNK: self.chunk_model.save_at(filename)
			if mode == POS: self.pos_model.save_at(filename)
			print("Model saved at "+ filename)
		except IOError:
			print("Cannot find or access location, model not saved")
	
	
	def tag(self, tokens, mode=CHUNK):
		"""Finds chunk or PoS tags for input list of tokens
		
		Uses the hmm object together with pos_model to generate a list 
		of pos tags from a list of tokens. If mode is CHUNK then the 
		list of pos tags is used together with chunking model in the 
		hmm object to generate a list of chunk tags.
		
		PRE: tokens is a list of strings representing tokens, mode if 
			specified is 0 (CHUNK) or 1 (POS)
		POST: returns list of chunk tags if mode is 0 (CHUNK) or list 
			of PoS tags if mode is 1 (POS)
		"""
		if self.pos_model:
			if self.chunk_model and mode == CHUNK:
				pos_nums = self.hmm.viterbi(tokens, self.pos_model)
				pos_tags = self.pos_model.converter.decode_tags(*pos_nums)
				chunk_nums = self.hmm.viterbi(pos_tags, self.chunk_model)
				chunk_tags = self.chunk_model.converter.decode_tags(*chunk_nums)
				return chunk_tags
			elif mode == CHUNK: print("No model for chunk tagging.")
			else:
				pos_tags = self.hmm.viterbi(tokens, self.pos_model)
				return pos_tags
		else: print("No model for part-of-speech pre-processing.")
	
	def tag_file(self, infile, outfile=None, mode=CHUNK):
		"""reads a .conll file and annotates with PoS or chunk tags
		
		Reads through a .conll file one sentence at a time, deliminated 
		by blank lines, and runs the tag function on each list of 
		tokens generated. The original lines are stored as strings and 
		the new annotation is added when at the appropriate column when 
		a sentence has been decoded, and are then outputted or written 
		to a file. 
		
		PRE: infile is filepath to a .conll file, mode is 0 (CHUNK) or 
			1 (POS), outfile if specified is the filepath for storing 
			generated annotated file
		POST: the contents of infile will be annotated with chunk or 
			PoS tags and outputtet either in terminal or at outfile
		"""
		parser = ConllParser()
		sentence, raw_lines = [], []
		line = infile.readline()
		while line:
			# if line isn't empty use it
			if line != "\n":
				token = parser.parse_line_TAG(line)
				sentence.append(token)
				raw_lines.append(line)
			# if end of sentence that is non empty
			elif len(sentence) > 0:
				out_sequence = self.tag(sentence, mode)
				if outfile:
					column = 5 if mode == CHUNK else 3
					for i, tag in enumerate(out_sequence):
						data = raw_lines[i].split("\t")
						data[column] = tag
						outfile.write("\t".join(data))
				else:
					print(out_sequence)
				sentence = []
			line = infile.readline()
				
	
	@staticmethod
	def test_UD(filename, filesize):
		parser = ConllParser()
		test_pos_model = "pos_model_UD_en.pickle"
		test_chunk_model = "chunk_model_UD_en.pickle"
		chunker = Chunker()
		chunker.load_model(test_pos_model, mode=POS)
		chunker.load_model(test_chunk_model, mode=CHUNK)
		correct_n, total_n, line_n, progress = 0, 0, 0, 0
		with open(filename, 'r') as inf:
			sentence, chunks = [], []
			line = inf.readline()
			while line:
				if line != "\n":
					token, chunk = parser.parse_line_EVAL(line)
					sentence.append(token)
					chunks.append(chunker.chunk_model.converter.convert_state(chunk))
				else:
					if len(sentence) > 0:
						guesses = chunker.tag(sentence, mode=CHUNK)
						for i in range(len(chunks)):
							if chunks[i] == guesses[i]: correct_n += 1
							total_n += 1
						sentence, chunks = [], []
						progress_value = int(line_n * 100 / filesize / 10)
						if progress_value > progress:
							progress = progress_value
							print( str(progress * 10) + "%")
							print("Accuracy:", correct_n / float(total_n))
				line_n += 1
				line = inf.readline()
		print("Final accuracy:", correct_n / float(total_n))

def init_args():
	parser = ArgumentParser(description="Simple bilingual monogram-based machine translator.")
	parser.add_argument("files", type=str, nargs='+', help="conll file(s) to process")
	parser.add_argument("-p", "--pos-model", type=str, nargs=1, help="specify a file to load for POS tagging model")
	parser.add_argument("-c", "--chunk-model", type=str, nargs=1, help="specify a file to load for POS tagging model")
	parser.add_argument("-o", "--output", type=str, nargs=1, help="specify file for output")
	parser.add_argument("-O", "--only-pos", action="store_true", help="only do POS preprocessing")
	parser.add_argument("-t", "--train", action="store_true", help="train models from files instead of loading")
	return parser.parse_args()
	
def main(args):
	chunker = Chunker()
	outfile = None
	mode = ''
	
	#determine mode
	if args.only_pos:
		mode = POS
	else:
		mode = CHUNK
	
	# determine if output should go to file
	if args.output and not args.train:
		try:
			outfile = open(args.output[0], 'w')
		except IOError as e:
			pring("Can't open", args.output[0])
			
	# if there is a pos model, load it
	if args.pos_model and not args.train:
		try:
			chunker.load_model(args.pos_model[0], mode=POS)
		except IOError:
			print("Cannot find or open", args.pos_model[0])					
			if outfile: outfile.close()
			return
		except pickle.UnpicklingError:
			print("Error while unpickling: is data corrupt?", args.pos_model[0])					
			if outfile: outfile.close()
			return
	
	# if there is a chunk model, load it
	if args.chunk_model and not args.train:
		try:
			chunker.load_model(args.chunk_model[0], mode=CHUNK)
		except IOError:
			print("Cannot find or open", args.pos_model[0])					
			if outfile:
				outfile.close()
			return
		except pickle.UnpicklingError:
			print("Error while unpickling: is data corrupt?", args.pos_model[0])					
			if outfile: outfile.close()
			return
		except:
			print("Unknown error while unpickling, is this a valid model file?", args.pos_model[0])					
			if outfile: outfile.close()
			return
		
	# determine if models should be trained rather than used
	if args.train:
		if args.pos_model: 
			pos_m = Model()
			pos_m.train(args.files[0])
			pos_m.save_at(args.pos_model)
		else:
			print("Use -p argument to specify outpath for trained POS model")
		if args.chunk_model: 
			chunk_m = Model()
			chunk_m.train(args.files[0], mode=CHUNK)
			chunk_m.save_at(args.chunk_model)
	# models are not to be trained, use them to tag!	
	else:
		for string in args.files:
			if string == args.output[0]:
				print(args.output[0], "is both input and output. ")
				break
			try:
				with open(string, 'r') as f:
					chunker.tag_file(f, outfile=outfile, mode=mode)
			except IOError as e:
				pring("Can't open", string)
	
	if outfile: outfile.close()

if __name__ == '__main__':
	args = init_args()
	main(args)
	#Chunker.test_UD("/home/peterpersson/lin503/projekt/universal_treebanks_v2.0/std/de-universal-test.conll", 17330)
	#Chunker.test_UD("/home/peterpersson/lin503/projekt/universal_treebanks_v1.0/en-universal-test.conll", 59100)

