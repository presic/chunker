#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#  converter.py
#  
#  this version is for the project
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

class Converter:
	"""Converts tags, tokens, and suffixes to ints with separate indexes
	
	Contains mappings of tags, tokens, and suffixes to integer 
	representations to be used in calculations. These mappings are 
	stored in one dict per type of symbol (i.e. one for tokens, one for 
	tags, etc.)"""
	
	def __init__(self):
		self.emission_index = {}
		self.state_index = {}
		self.suffix_index = {}
	
	def convert_emission(self, e):
		return self.emission_index.setdefault(e, len(self.emission_index))
	
	def convert_state(self, q):
		return self.state_index.setdefault(q, len(self.state_index))
	
	def convert_suffix(self, s):
		return self.state_index.setdefault(q, len(self.suffix_index))
	
	def _decode(self, e, index_list):
		"""Looks up and returns index e in provided list."""
		return index_list[e]
		
	def get_states(self):
		"""Returns a sorted list of states"""
		return sorted(list(self.state_index.keys()), 
					key=lambda x: self.state_index[x])
	
	def get_state_N(self):
		return len(self.state_index)
	
	def get_emissions(self):
		"""Returns a sorted list of emissions"""
		return sorted(list(self.emission_index.keys()), 
					key=lambda x: self.emission_index[x])
	
	def get_emission_N(self):
		return len(self.emission_index)
		
	def get_suffixes(self):
		"""Returns a sorted list of suffixes"""
		return sorted(list(self.suffix_index.keys()), 
						key=lambda x: self.state_index[x])
	
	def get_suffix_N(self):
		return len(self.suffix_index)
	
	def convert_tokens(self, *args):
		"""Converts tokens passed as string arguments to ints
		
		Returns a list of ints."""
		return [self.convert_emission(word) for word in args]
	
	def convert_tags(self, *args):
		"""Converts tags passed as string arguments to ints
		
		Returns a list of ints."""
		return [self.convert_state(tag) for tag in args]
	
	def convert_both(self, *args):
		"""Converts (token,tag) containing strings to corresponding ints
		
		Returns a list of (token,tag) containing ints"""
		return [(self.convert_emission(entry[0]), 
				 self.convert_state(entry[1])) + entry[2:] 
						for entry in args]
	
	def decode_both(self, *args):
		"""Decodes (token, tag) containing ints passed as arguments
		
		Returns a list of (token,tag) containing strings"""
		emission_list = self.get_emissions()
		state_list = self.get_states()
		return [(self._decode(entry[0], emission_list), 
				 self._decode(entry[1], state_list)) for entry in args]
	
	def decode_tags(self, *args):
		"""Decodes int representations of tags passed as arguments
		
		Returns a list of strings."""
		state_list = self.get_states()
		return [self._decode(tag, state_list) for tag in args]
	
	def decode_tokens(self, *args):
		"""Decodes int representations of tokens passed as arguments
		
		Returns a list of strings."""
		emission_list = self.get_emissions()
		return [self._decode(word, emission_list) for word in args]
	
	
	@staticmethod
	def test():
		train_sentences = ["Hej/INJ ,/PUN hur/WHW mår/VRB du/PRN ?/PUN","Jag/PRN mår/VRB bra/ADV ,/PUN tack/PRT !/PUN","Skitgubbe/NN ,/PUN det/PRN var/VRB inte/ADV dig/PRN jag/PRN pratade/VRB med/PP !/PUN","Du/PRN kan/VRB va/VRB en/ART skitgubbe/NN själv/PRN och/KNJ se/VRB hur/WHW lätt/JJ det/PRN är/VRB ,/PUN pojkspoling/NN !/PUN"]
		test_sentences = ["Hej , hur mår du ?","Jag mår bra , tack !","Skitgubbe , det var inte dig jag pratade med !","Du kan va en skitgubbe själv och se hur lätt det är , pojkspoling !"]
		c = Converter()
		c.train([[entry.split('/') for entry in sentence.split()] for sentence in train_sentences])
		print([c.convert_sentence(sentence.split()) for sentence in test_sentences])
		print('emissions:',c.get_emission_N(), ':', c.get_emissions())
		print('states:',c.get_state_N(), ':', c.get_states())

if __name__ == '__main__':
	Converter.test()
