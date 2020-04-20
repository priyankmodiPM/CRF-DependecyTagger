import torch
import pickle
import unicodedata
import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm
from parameters import *
from bpemb import BPEmb

random_seed = 42
np.random.seed(random_seed)

class Model_Data:
	def __init__(self, name):
		self.name = name
		self.all_words = []
		self.all_tags = []
		self.all_langs = []
		self.lang_vocab = {}
		self.joint_embeds = {}
		self.max_length = 0
		self.train_data, self.train_targets, self.valid_data, self.valid_targets = self.read_data()
		self.test_data, self.test_targets = self.read_test_data()
		self.word2index , self.hi_emb = self.build_vocab()
		self.n_words = len(self.word2index.keys())
		
		self.index2word = {}
		for word,index in self.word2index.items():
			self.index2word[index] = word
		print(self.index2word)
		self.char2index = self.char_mapping(self.all_words)
		self.index2char = {}
		for char, index in self.char2index.items():
			self.index2char[index] = char
		
		self.tag2index = self.tag_mapping(self.all_tags)
		self.index2tag = {}
		print(self.tag2index)
		for index, tag in self.tag2index.items():
			self.index2tag[index] = tag

		# self.lang2index = self.tag_mapping(self.all_langs)
		# print(self.lang2index)
		# self.index2lang = {}
		# for index, lang in self.index2lang.items():
		# 	self.index2lang[index] = lang

	def char_mapping(self, all_words):
		"""
		Create a dictionary and mapping of characters.
		"""
		print("[INFO] -> Indexing Characters.")
		char2index = {}
		chars = "".join(word for word in all_words)
		chars = list(set(list(chars)))
		for index, char in enumerate(chars):
			char2index[char] = index
		return char2index

	def tag_mapping(self, all_tags):
		tag2index = {}
		tags = list(set(all_tags))
		for index, tag in enumerate(tags):
			tag2index[tag] = index
		return tag2index
   	
	def get_item(self, emb_model, word):
		ids = emb_model.encode_ids([word])
		embs = emb_model.vectors[ids]
		return np.mean(embs,axis=0)

	def get_emb(self, model, lang):
		muse_dict = {}
		counter = 0
		new_model = BPEmb(lang="hi", dim=300, vs=200000)
		print("[INFO] -> Loading Embeddings for :", lang)
		muse_dict[PAD_WORD] = np.array(([0 for i in range(300)])).astype(np.float)
		try:
			for word in self.lang_vocab[lang]:
				try:
					# print(word)
					muse_dict[word] = self.get_item(new_model, word)
				except KeyError:
					print(word)
					counter +=1
		except KeyError:
			print("No Embeds For Lang :", lang)
			return muse_dict
		print("OOV Words : ", counter)
		return muse_dict

	def combine_embeds(self, eng, ita, esp, fra):
		muse_joint = {}
		eng_keys = list(eng.keys())
		ita_keys = list(ita.keys())
		esp_keys = list(esp.keys())
		fra_keys = list(fra.keys())

		all_keys = eng_keys + ita_keys + esp_keys + fra_keys
		all_keys = list(set(all_keys))

		counter = 0

		for word in all_keys:
			en, it, fr, sp = 0, 0, 0, 0

			if word in eng_keys:
				muse_joint[word] = eng[word]
				en = 1

			if word in ita_keys:
				if word in muse_joint:
					muse_joint[word] = muse_joint[word] + ita[word]
				else:
					muse_joint[word] = ita[word]
				it = 1
			
			if word in fra_keys:
				if word in muse_joint:
					muse_joint[word] = muse_joint[word] + fra[word]
				else:
					muse_joint[word] = fra[word]
				fr = 1
			
			if word in esp_keys:
				if word in muse_joint:
					muse_joint[word] = muse_joint[word] + esp[word]
				else:
					muse_joint[word] = esp[word]
				sp = 1

			total = en + it + fr + sp

			muse_joint[word] = muse_joint[word]/total

			if total > 1:
				counter += 1

		print(counter)
		return muse_joint

	def write_pickle(self, data, filename):
		with open('../embeds/'+filename, 'wb') as handle:
			pickle.dump(data, handle)

	def read_pickle(self, filename):
		with open('../embeds/'+filename, 'rb') as handle:
			b = pickle.load(handle)
		return b

	def build_vocab(self):
		'''
		print("[INFO] -> Reading Embeddings files.")
		print("[INFO] -> Reading English Embeddings")
		en_model = KeyedVectors.load_word2vec_format('../embeds/MUSE/wiki.multi.en.vec')
		print("[INFO] -> Reading Italian Embeddings")
		it_model = KeyedVectors.load_word2vec_format('../embeds/MUSE/wiki.multi.it.vec')
		print("[INFO] -> Reading French Embeddings")
		fr_model = KeyedVectors.load_word2vec_format('../embeds/MUSE/wiki.multi.fr.vec')
		print("[INFO] -> Reading Spanish Embeddings")
		es_model = KeyedVectors.load_word2vec_format('../embeds/MUSE/wiki.multi.es.vec')

		print("[INFO] -> Normalizing Embeddings")
		it_model.init_sims()
		en_model.init_sims()
		es_model.init_sims()
		fr_model.init_sims()
		
		muse_eng = self.get_emb(en_model, "english")
		muse_ita = self.get_emb(it_model, "italian")
		muse_esp = self.get_emb(es_model, "spanish")
		muse_fra = self.get_emb(fr_model, "french")

		self.joint_embeds = self.combine_embeds(muse_eng, muse_ita, muse_esp, muse_fra)


		self.write_pickle(muse_eng, "muse_eng")
		self.write_pickle(muse_ita, "muse_ita")
		self.write_pickle(muse_esp, "muse_esp")
		self.write_pickle(muse_fra, "muse_fra")
		self.write_pickle(self.joint_embeds, "muse_joint")

		print("Written All Files")
		exit(0)
		'''
		
		# print("[INFO] -> Reading Kannada Embeddings")
		print("[INFO] -> Reading Hindi Embeddings")
		# kannada_model = KeyedVectors.load_word2vec_format('../embeds/cc.kn.300.vec')
		# kannada_model.init_sims()
		# kannada_model = ""
		hi_model = ""
		# kannada_emb = self.get_emb(kannada_model, "kannada")
		hi_emb = self.get_emb(hi_model, "hi")

		# muse_esp = self.read_pickle("muse_esp")
		# muse_fra = self.read_pickle("muse_fra")
		# muse_ita = self.read_pickle("muse_ita")
		# self.joint_embeds = self.read_pickle('muse_joint')
		
		# words = list(set(list(kannada_emb.keys()))) #+ list(muse_ita.keys()) + list(muse_fra.keys()) + list(muse_esp.keys())))
		words = list(set(list(hi_emb.keys()))) #+ list(muse_ita.keys()) + list(muse_fra.keys()) + list(muse_esp.keys())))
		word2index = {}
		# word2index[PAD_WORD] = PAD
		ind = 0
		for word in words:
			if word == 'unk':
				print(word)
				continue
			word2index[word] = ind
			ind += 1
		
		word2index['unk'] = ind
		# return word2index, kannada_emb #, muse_ita, muse_esp, muse_fra
		return word2index, hi_emb #, muse_ita, muse_esp, muse_fra

	def unicode_to_ascii(self, s):
		return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

	def split_train_valid(self, all_pairs):

		print("[INFO] -> Splitting data into train/valid.")
		validation_split = 0.1
		dataset_size = len(all_pairs)
		indices = list(range(dataset_size))
		split = int(np.floor(validation_split * dataset_size))
		np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]

		train_data = []
		valid_data = []
		train_targets = []
		valid_targets = []

		for ind in tqdm(train_indices,desc = "[INFO] -> Prep Training data") :
			train_data.append([all_pairs[ind][0],all_pairs[ind][2]])
			train_targets.append(all_pairs[ind][1])

		for ind in tqdm(val_indices,desc = "[INFO] -> Prep Valid data") :
			valid_data.append([all_pairs[ind][0],all_pairs[ind][2]])
			valid_targets.append(all_pairs[ind][1])

		return train_data, train_targets, valid_data, valid_targets

	def read_lang_file(self, lang):
		all_pairs = []
		current_sentence = []
		current_tags = []
		count = 0
		with open("./data/preprocessed_train2.dat",'r') as f:
			for line in tqdm(f.readlines(),desc="[INFO] -> Reading data."):	
				if len(line.split()) == 0:
					if len(current_sentence) > self.max_length:
						self.max_length = len(current_sentence)
					if len(current_sentence) != len(current_tags):
						print("[INFO] -> ERROR, len of sentence != len of tags.")
						exit(0)
					sent = " ".join(current_sentence)
					label = " ".join(current_tags)
					all_pairs.append([sent,label,lang])
					current_sentence = []
					current_tags = []
				# elif line.split()[0] == "<s>":
				# 	count+=1
				# elif line.split()[0] == "":
				# 	pass
				else:
					if line.split()[1] == "Temp":
						print("aa")
						print(line.split()[0])

					word = self.unicode_to_ascii(line.split()[0].lower())
					# word = str(line.split()[0])
					self.all_words.append(word)
					try:
						self.lang_vocab[lang].append(word)
					except KeyError:
						print("error")
						print(line)
						# print(self.lang_vocab[lang])
						self.lang_vocab[lang] = [word]
						print(self.lang_vocab[lang])

					try:
						tag = line.split()[1]
						self.all_tags.append(tag)
						current_sentence.append(word)
						current_tags.append(tag)
						self.all_langs.append(lang)
						# print(word)
						# print(count)
		

					except IndexError:
						print("assad")
						print(line)
						print(lang)
						exit()
						print(lang)
					
		# self.lang_vocab[lang].append('unk')
		self.lang_vocab[lang] = list(set(self.lang_vocab[lang]))
		print("[INFO] -> Vocab size of : ", lang, " is : ", len(self.lang_vocab[lang]))
		
		if len(current_sentence) > self.max_length:
			self.max_length = len(current_sentence)
		if len(current_sentence) != len(current_tags):
			print("[INFO] -> ERROR, len of sentence != len of tags.")
			exit(0)

		sent = " ".join(current_sentence)
		label = " ".join(current_tags)
		all_pairs.append([sent,label,lang])
		return all_pairs

	def read_data(self):

		# all_pairs = self.read_lang_file("kannada")
		all_pairs = self.read_lang_file("hi")
		# print(len(self.lang_vocab))
		# it_pairs = self.read_lang_file("italian")
		# esp_pairs = self.read_lang_file("spanish")
		#fra_pairs = self.read_lang_file("french")
		# all_pairs = eng_pairs + it_pairs + esp_pairs

		self.all_words = list(set(self.all_words))
		self.all_tags = list(set(self.all_tags))
		# self.all_langs = list(set(self.all_langs))
	
		train_data, train_targets, valid_data, valid_targets = self.split_train_valid(all_pairs)
		return train_data, train_targets, valid_data, valid_targets

	def read_lang_test(self, lang):
		all_pairs = []
		current_sentence = []
		current_tags = []
		with open("./data/preprocessed_test1.dat",'r') as f:
			for line in tqdm(f.readlines(),desc="[INFO] -> Reading data."):
				if len(line.split()) == 0:
					if len(current_sentence) > self.max_length:
						self.max_length = len(current_sentence)
					if len(current_sentence) != len(current_tags):
						print("[INFO] -> ERROR, len of sentence != len of tags.")
						exit(0)
					sent = " ".join(current_sentence)
					label = " ".join(current_tags)
					all_pairs.append([sent,label,lang])
					current_sentence = []
					current_tags = []
				elif line.split()[0] == "<s>" or line.split()[0] == "</s>":
					pass
				else:
					word = self.unicode_to_ascii(line.split()[0].lower())
					self.all_words.append(word)
					try:
						self.lang_vocab[lang].append(word)
					except KeyError:
						print("error")
						self.lang_vocab[lang] = [word]
					tag = line.split()[1]
					# if tag == 'TEMP'
					self.all_tags.append(tag)
					current_sentence.append(word)
					current_tags.append(tag)
		
		self.lang_vocab[lang].append('unk')
		self.lang_vocab[lang] = list(set(self.lang_vocab[lang]))
		print("[INFO] -> Vocab size of : ", lang, " is : ", len(self.lang_vocab[lang]))
		
		if len(current_sentence) > self.max_length:
			self.max_length = len(current_sentence)
		if len(current_sentence) != len(current_tags):
			print("[INFO] -> ERROR, len of sentence != len of tags.")
			exit(0)

		sent = " ".join(current_sentence)
		label = " ".join(current_tags)
		all_pairs.append([sent,label,lang])
		return all_pairs

	def read_test_data(self):
		#eng_pairs = self.read_lang_test("english")
		#it_pairs = self.read_lang_test("italian")
		#esp_pairs = self.read_lang_test("spanish")
		hin_pairs = self.read_lang_test("hi")
		all_pairs = hin_pairs

		self.all_tags = list(set(self.all_tags))
		self.all_langs = list(set(self.all_langs))

		indices = list(range(len(all_pairs)))
		np.random.shuffle(indices)

		test_data = []
		test_targets = []

		for ind in tqdm(indices,desc = "[INFO] -> Prep Test data") :
			test_data.append([all_pairs[ind][0],all_pairs[ind][2]])
			test_targets.append(all_pairs[ind][1])

		return test_data, test_targets

def main():
	
	model_name = "bpe_hindi_train_3.pt"
	model_data = Model_Data("Datset data")
	print("[INFO] -> Saving file.")
	torch.save(model_data, "./data_models/" + model_name)
	print(model_name)

if __name__ == '__main__':
	main()
