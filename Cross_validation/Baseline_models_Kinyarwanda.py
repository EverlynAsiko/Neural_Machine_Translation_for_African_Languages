#!/usr/bin/env python
# coding: utf-8

# # Kinyarwanda Baseline Models 
# 

# #### Setting up locations and libraries

# In[7]:


# Importing needed libraries for preprocessing and visualization
import numpy as np
import pandas as pd
import os


# In[4]:


#@title Default title text
# Install Pytorch with GPU support v1.8.0.
#os.system(' pip install torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html')
#os.system(' pip install -U torchtext==0.9.0')


# In[5]:


# Filtering warnings
import warnings
warnings.filterwarnings('ignore')


# In[28]:


# Setting source and target languages
source_language = "en"
target_language = "rw"

os.environ["src"] = source_language 
os.environ["tgt"] = target_language


# In[9]:

#os.system('python3 -m pip install -U pip')

#os.system(' rm -rf joeynmt')
#os.system(' git clone https://github.com/joeynmt/joeynmt.git')
#os.system(' cd joeynmt; pip3 install .')
#os.system(' cd joeynmt; pip install .')
#os.system(' pip3 install subword-nmt')
#os.system(' pip install subword-nmt')


# # Getting Data

# ### Data preprocessing

# In[10]:


rwa = pd.read_csv("Kinyarwanda.csv")
rwa.head(3)


# In[11]:


# drop duplicate translations
df_pp = rwa.drop_duplicates()

# drop conflicting translations
df_pp.drop_duplicates(subset='source_sentence', inplace=True)
df_pp.drop_duplicates(subset='target_sentence', inplace=True)

# Shuffle the data to remove bias in dev set selection.
df_pp = df_pp.sample(frac=1, random_state=42).reset_index(drop=True)


# In[12]:


# reset the index of the training set after previous filtering
df_pp.reset_index(drop=False, inplace=True)


# In[13]:


df_pp.dropna(inplace=True)


# In[14]:


df_pp.isna().sum()


# In[15]:

#os.system(' pip3 install -U scikit-learn')
#os.system(' pip install -U scikit-learn')
from sklearn.model_selection import KFold


# In[16]:


kfold = KFold(100, True, 1)


# In[18]:


# reset the index of the training set after previous filtering
df_pp.reset_index(drop=True, inplace=True)


# In[19]:


# enumerate splits
for train, test in kfold.split(df_pp):
    train = train.tolist()
    test = test.tolist()
    #print(df_pp.loc[train],df_pp.loc[test])


# In[25]:


def split_srctgt(train, test):
  # Splitting train and validation set
  num_valid = 100

  dev = train.tail(num_valid) 
  stripped = train.drop(train.tail(num_valid).index)

  # Creating files: Train
  with open("train."+source_language, "w") as src_file, open("train."+target_language, "w") as trg_file:
    for index, row in stripped.iterrows():
      src_file.write(row["source_sentence"]+"\n")
      trg_file.write(row["target_sentence"]+"\n")

  # Dev   
  with open("dev."+source_language, "w") as src_file, open("dev."+target_language, "w") as trg_file:
    for index, row in dev.iterrows():
      src_file.write(row["source_sentence"]+"\n")
      trg_file.write(row["target_sentence"]+"\n")

  # Test
  with open("test."+source_language, "w") as src_file, open("test."+target_language, "w") as trg_file:
    for index, row in test.iterrows():
      src_file.write(row["source_sentence"]+"\n")
      trg_file.write(row["target_sentence"]+"\n")


# In[29]:


def generating_BPE():
  # Apply BPE splits to the development and test data.
  os.system(' subword-nmt learn-joint-bpe-and-vocab --input train.$src train.$tgt -s 4000 -o bpe.codes.4000 --write-vocabulary vocab.$src vocab.$tgt')

  # Apply BPE splits to the development and test data.
  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$src < train.$src > train.bpe.$src')
  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$tgt < train.$tgt > train.bpe.$tgt')

  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$src < dev.$src > dev.bpe.$src')
  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$tgt < dev.$tgt > dev.bpe.$tgt')
  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$src < test.$src > test.bpe.$src')
  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$tgt < test.$tgt > test.bpe.$tgt')

  # Create that vocab using build_vocab
  os.system(' chmod 755 joeynmt/scripts/build_vocab.py')
  os.system(' joeynmt/scripts/build_vocab.py train.bpe.$src train.bpe.$tgt --output_path vocab.txt')


# # Modeling

# ## Kinyarwanda

# In[23]:


def run_model():
    path = os.getcwd() 
    name = '%s%s' % (target_language, source_language)
    
    # Create the config
    config = """
    name: "{target_language}{source_language}_reverse_transformer"

    data:
      src: "{target_language}"
      trg: "{source_language}"
      train: "{path}/train.bpe"
      dev:   "{path}/dev.bpe"
      test:  "{path}/test.bpe"
      level: "bpe"
      lowercase: False
      max_sent_length: 100
      src_vocab: "{path}/vocab.txt"
      trg_vocab: "{path}/vocab.txt"

    testing:
      beam_size: 5
      alpha: 1.0

    training:
      #load_model: "{path}/models/{name}_transformer/1.ckpt" # if uncommented, load a pre-trained model from this checkpoint
      random_seed: 42
      optimizer: "adam"
      normalization: "tokens"
      adam_betas: [0.9, 0.999] 
      scheduling: "plateau"           # TODO: try switching from plateau to Noam scheduling
      patience: 5                     # For plateau: decrease learning rate by decrease_factor if validation score has not improved for this many validation rounds.
      learning_rate_factor: 0.5       # factor for Noam scheduler (used with Transformer)
      learning_rate_warmup: 1000      # warmup steps for Noam scheduler (used with Transformer)
      decrease_factor: 0.7
      loss: "crossentropy"
      learning_rate: 0.0003
      learning_rate_min: 0.00000001
      weight_decay: 0.0
      label_smoothing: 0.1
      batch_size: 4096
      batch_type: "token"
      eval_batch_size: 1000
      eval_batch_type: "token"
      batch_multiplier: 1
      early_stopping_metric: "ppl"
      epochs: 30                  # TODO: Decrease for when playing around and checking of working. Around 30 is sufficient to check if its working at all
      validation_freq: 4000         # TODO: Set to at least once per epoch.
      logging_freq: 200
      eval_metric: "bleu"
      model_dir: "models/{name}_reverse_transformer"
      overwrite: True              # TODO: Set to True if you want to overwrite possibly existing models. 
      shuffle: True
      use_cuda: True
      max_output_length: 100
      print_valid_sents: [0, 1, 2, 3]
      keep_last_ckpts: 3

    model:
      initializer: "xavier"
      bias_initializer: "zeros"
      init_gain: 1.0
      embed_initializer: "xavier"
      embed_init_gain: 1.0
      tied_embeddings: True
      tied_softmax: True
      encoder:
          type: "transformer"
          num_layers: 6
          num_heads: 4             # TODO: Increase to 8 for larger data.
          embeddings:
              embedding_dim: 256   # TODO: Increase to 512 for larger data.
              scale: True
              dropout: 0.2
          # typically ff_size = 4 x hidden_size
          hidden_size: 256         # TODO: Increase to 512 for larger data.
          ff_size: 1024            # TODO: Increase to 2048 for larger data.
          dropout: 0.3
      decoder:
          type: "transformer"
          num_layers: 6
          num_heads: 4              # TODO: Increase to 8 for larger data.
          embeddings:
              embedding_dim: 256    # TODO: Increase to 512 for larger data.
              scale: True
              dropout: 0.2
          # typically ff_size = 4 x hidden_size
          hidden_size: 256         # TODO: Increase to 512 for larger data.
          ff_size: 1024            # TODO: Increase to 2048 for larger data.
          dropout: 0.3
    """.format(name=name, path=path, source_language=source_language, target_language=target_language)
    with open("joeynmt/configs/transformer_reverse_{name}.yaml".format(name=name),'w') as f:
      f.write(config)

    # Train the model
    os.system('cd joeynmt; python3 -m joeynmt train configs/transformer_reverse_$tgt$src.yaml')
    os.system("cd joeynmt; python3 -m joeynmt test 'models/rwen_reverse_transformer/config.yaml'")


# In[30]:


result = []
count = 0
for train, test in kfold.split(df_pp):
    count +=1
    print(count)
    if count == 7:
        break
    train = train.tolist()
    test = test.tolist()
    split_srctgt(df_pp.loc[train],df_pp.loc[test])
    generating_BPE()
    print('Done with BPE')
    run_model()
    
    fff   = open('joeynmt/models/rwen_reverse_transformer/test.log', 'r')   
    X = fff.readlines()    # Reading to a list 
    fff.close()
    
    result.append(X)


# In[ ]:


with open('result_kinyarwanda.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % place for place in result)
