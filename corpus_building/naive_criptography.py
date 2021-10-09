# -*- coding: utf-8 -*-
# This program was made by Felipe Serras as part of his Master's degree,
# under the guidance of Prof. Marcelo Finger. All rights reserved.
# We tried to make explicit all our references and all the works on which ours is based.
# Please contact us if you encounter any problems in this regard.
# If not stated otherwise, this software is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Naive Criptography
A module that defines simple encryption procedures for the management of the produced corpus .

Made based on this reference:
https://stackoverflow.com/questions/12524994/encrypt-decrypt-using-pycrypto-aes-256
"""

from Crypto.Cipher import AES
import hashlib


class Naive_Criptography:
    """Class that encapsulates basic encryption procedures
    """

    def __init__(self):
        self.IV = b')\xcbi\xf9\xb3~\xdb\xedU\xb1h\xf2pDG\xae'
        self.mode = AES.MODE_CFB

    def naive_encrypt(self, password, text):
        """Procedure to encrypt a piece of text.

        Args:
            password (str): password under which the text will be encrypted 
            text (str): text to be encrypted

        Returns:
            [str]: encrypted text
        """
        password = password.encode('utf-8')
        key = hashlib.sha256(password).digest()
        text.encode('utf-8')
        encryptor = AES.new(key, self.mode, IV=self.IV)
        ciphertext = encryptor.encrypt(text)
        return ciphertext

    def naive_decrypt(self, password, ciphertext):
        """Procedure to decrypt a piece of text.

        Args:
            password (str): password to decrypt the text
            ciphertext (str): encrypted text to be decrypted

        Returns:
            str: decrypted text
        """
        password = password.encode('utf-8')
        key = hashlib.sha256(password).digest()
        decryptor = AES.new(key, self.mode, IV=self.IV)
        return decryptor.decrypt(ciphertext).decode('utf-8')

    def naive_corpus_encryption(self, password, corpus):
        """Procedure to encrypt an entire corpus

        Args:
            password (str): password under which the corpus content will be encrypted 
            corpus (list): textual corpus to be encrypted

        Returns:
            list: encrypted corpus
        """
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                if type(corpus[i][j]) != list:
                    corpus[i][j] = self.naive_encrypt(password, corpus[i][j])
                else:
                    for k in range(len(corpus[i][j])):
                        corpus[i][j][k] = self.naive_encrypt(
                            password, corpus[i][j][k])
        return corpus

    def naive_corpus_decryption(self, password, corpus):
        """Procedure to decrypt an entire corpus

        Args:
            password (str): password to decrypt the corpus content
            corpus (list): encrypted corpus

        Returns:
            list: decrypted corpus
        """
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                if type(corpus[i][j]) != list:
                    corpus[i][j] = self.naive_decrypt(password, corpus[i][j])
                else:
                    for k in range(len(corpus[i][j])):
                        corpus[i][j][k] = self.naive_decrypt(
                            password, corpus[i][j][k])
        return corpus
