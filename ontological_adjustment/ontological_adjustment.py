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

"""Ontological Adjustment
This script adjusts the data class system, returning a new ontologically adjusted corpus.

This ontological adjustment consists of pre-processing of classes, filtering by frequency,
construction of hierarchy by co-occurrence, agglutination of classes with low occurrence
and grouping of classes, in order to generate a hierarchical structure of classes and a
corresponding corpus, classified accordingly.
"""

from naive_criptography import *
import pickle
from namedlist import namedlist
from getpass import getpass
import numpy as np
from tqdm import tqdm
import re
import sys
import matplotlib.pyplot as plt
import nltk
import unidecode
import copy
from anytree import Node, RenderTree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import norm as sp_norm
from sklearn.cluster import Birch

# Global data structures:

CORPUS_FILE_NAME = 'koll_corpus_01.pkl'
FIELDS_FILE_NAME = 'koll_fields_01.pkl'

LEGAL_MULTIWORD_EXPRESSIONS = ['a_quo', 'abolitio_criminis', 'ad_quem', 'advocacia-geral_da_uni??o', 'amicus_curiae',
                               'bis_in_idem', 'caso_fortuito', 'causa_de_pedir', 'cl??usula_p??trea', 'coisa_julgada',
                               'ex_officio', 'ex_nunc', 'ex_tunc', 'extradi????o', 'for??a_maior', 'habeas_corpus',
                               'habeas_data', 'iniciativa_de_lei', 'kompetenz-kompetenz', 'neg??cio_jur??dico',
                               'nexo_causal', 'ordem_dos_advogados_do_brasil', 'partido_pol??tico', 'queixa_crime',
                               'quinto_constitucional', 'transitar_em_julgado', 'distrito_policial',
                               'acessorium_sequitur_principale', 'ad_corpus', 'ad_impossibilia_nemo_tenetur',
                               'ad_mensuram', 'advocacia_geral_da_uni??o', 'ad_corpus', 'causa_petendi', 'amicus_curiae',
                               'ano_novo', 'a-quo', 'abolitio-criminis', 'ad-quem', 'advocacia-geral-da-uni??o',
                               'amicus-curiae', 'bis-in-idem', 'caso-fortuito', 'causa-de-pedir', 'cl??usula-p??trea',
                               'coisa-julgada', 'de-of??cio', 'ex-officio', 'ex-nunc', 'ex-tunc', 'extradi????o',
                               'for??a-maior', 'habeas-corpus', 'habeas-data', 'iniciativa-de-lei',
                               'kompetenz-kompetenz', 'neg??cio-jur??dico', 'nexo-causal',
                               'ordem-dos-advogados-do-brasil', 'partido-pol??tico', 'queixa-crime',
                               'quinto-constitucional', 'transitar-em-julgado', 'distrito-policial',
                               'acessorium-sequitur-principale', 'ad-corpus', 'ad-impossibilia-nemo-tenetur',
                               'ad-mensuram', 'advocacia-geral-da-uni??o', 'ad-corpus', 'causa-petendi', 'amicus-curiae',
                               'ano-novo']
GENERAL_MULTIWORD_EXPRESSIONS = ['abaixo_assinado', 'abaixo_assinados', 'aca_de_a??ucares', 'alto_falante',
                                 'alto_falantes', 'anos_luz', 'anti_retrovirais', 'anti_retroviral', 'ar_condicionado',
                                 'ar_condicionados', 'arco_??ris', 'ares_condicionados', 'atividade_fim',
                                 'atividades_fim', 'auto_imune', 'auto_imunes', 'banho_maria', 'batata_doce',
                                 'batatas_doce', 'bate_boca', 'bate_papo', 'bate_papos', 'beija_flor', 'beija_flores',
                                 'bem_estar', 'bem_sucedida', 'bem_sucedidas', 'bem_sucedido', 'bem_sucedidos',
                                 'bem_vinda', 'bem_vindas', 'bem_vindo', 'bem_vindos', 'boa_f??', 'boas_vindas',
                                 'boca_de_urna', 'caixa_preta', 'caixa_pretas', 'caixas_preta', 'caixas_pretas',
                                 'cana_de_a????car', 'canas_de_a????car', 'capim_elefante', 'carro_chefe', 'carro_forte',
                                 'carros_chefe', 'carros_forte', 'case_control', 'cd_rom', 'cd_roms', 'censo_designada',
                                 'censo_designadas', 'censo_designado', 'censo_designados', 'coca_cola', 'coca_colas',
                                 'contra_ataque', 'contra_ataques', 'cor_de_rosa', 'corpo_de_prova', 'corpo_de_provas',
                                 'corpos_de_provas', 'cross_sectional', 'curta_metragem', 'curta_metragens',
                                 'curto_circuito', 'curto_circuitos', 'c??lula_tronco', 'c??lulas_tronco', 'data_base',
                                 'data_bases', 'decreto_lei', 'decreto_leis', 'decretos_lei', 'decretos_leis',
                                 'desvio_padr??o', 'desvio_padr??es', 'desvios_padr??o', 'dia_a_dia', 'dona_de_casa',
                                 'donas_de_casa', 'double_blind', 'duplo_cego', 'e_mail', 'e_mails', 'ene_??sima',
                                 'ene_??simas', 'erva_mate', 'ervas_mate', 'follow_up', 'free_lance', 'free_lancers',
                                 'free_lances', 'gr??_bretanha', 'guarda_costas', 'guarda_roupa', 'guarda_roupas',
                                 'high_risk', 'hip_hop', 'i_??sima', 'i_??simas', 'infra_estrutura', 'infra_estruturas',
                                 'inter_relacionada', 'inter_relacionadas', 'inter_relacionado', 'inter_relacionados',
                                 'inter_relacionamentos', 'inter_rela????es', 'key_word', 'key_words', 'know_how',
                                 'long_term', 'longa_metragem', 'longa_metragens', 'lua_de_mel', 'luas_de_mel',
                                 'lugar_comum', 'l??ngua_alvo', 'l??nguas_alvo', 'mais_valia', 'mal_entendido',
                                 'mal_estar', 'mata_mata', 'mat??ria_prima', 'mat??ria_primas', 'mau_estar',
                                 'meia_atacante', 'meia_atacantes', 'meia_idade', 'meia_noite', 'meia_vida',
                                 'meia_vidas', 'meio_ambiente', 'meio_campo', 'meio_campos', 'meio_dia', 'meio_termo',
                                 'meio_termos', 'meios_campo', 'meios_termos', 'mercedes_benz', 'meta_analysis',
                                 'meu_entendido', 'm??_f??', 'm??o_de_obra', 'n_??sima', 'n_??simas', 'obra_prima',
                                 'obras_primas', 'oitava_de_final', 'oitavas_de_final', 'on_line', 'p_valor',
                                 'padr??o_ouro', 'padr??es_ouro', 'palavra_chave', 'palavras_chave', 'palavras_chaves',
                                 'pan_americana', 'pan_americanas', 'pan_americanos', 'pau_brasil', 'paus_brasil',
                                 'paus_brasis', 'pay_per_view', 'pay_per_views', 'peso_pesado', 'pesos_pesados',
                                 'pesquisa_a????o', 'pesquisas_a????o', 'placebo_controlled', 'population_based',
                                 'popula????o_alvo', 'popula????es_alvo', 'porta_avi??es', 'porta_enxerto', 'porta_enxertos',
                                 'porta_malas', 'porta_voz', 'porta_vozes', 'post_mortem', 'primeira_dama',
                                 'primeira_damas', 'primeiras_damas', 'pronto_atendimento', 'pronto_atendimentos',
                                 'pronto_socorro', 'pronto_socorros', 'prontos_atendimentos', 'prontos_socorros',
                                 'pr??_frontal', 'pr??_natais', 'pr??_natal', 'pr??_requisito', 'pr??_requisitos',
                                 'p??ra_quedas', 'p??r_do_sol', 'quadro_negro', 'quadros_negros', 'quarta_de_final',
                                 'quarta_feira', 'quartas_de_final', 'quartas_feiras', 'quarteis_generais',
                                 'quartel_general', 'quebra_cabe??a', 'quebra_cabe??as', 'qui_quadrado', 'quinta_feira',
                                 'quintas_feiras', 'raio_x', 'raios_x', 'r??_tim_bum', 'salva_vidas',
                                 'secret??rio_executivo', 'secret??rios_executivos', 'segunda_feira', 'segundas_feiras',
                                 'sexta_feira', 'sexta_feiras', 'sextas_feiras', 'short_term', 'springer_verlag',
                                 'super_homem', 'ter??a_feira', 'ter??as_feiras', 'todo_poderoso', 'ultra_som',
                                 'ultra_sonografia', 'ultra_sonografias', 'ultra_sonogr??fico', 'ultra_sonogr??ficos',
                                 'ultra_sons', 'ultra_s??nica', 'ultra_s??nico', 'ultra_s??nicos', 'ultra???s??nicas',
                                 'vice_versa', 'villa_lobos', 'vis_??_vis', 'x_man', 'x_men', 'x_ray', 'x_rays',
                                 'abaixo-assinado', 'abaixo-assinados', 'aca-de-a??ucares', 'alto-falante',
                                 'alto-falantes', 'anos-luz', 'anti-retrovirais', 'anti-retroviral', 'ar-condicionado',
                                 'ar-condicionados', 'arco-??ris', 'ares-condicionados', 'atividade-fim',
                                 'atividades-fim', 'auto-imune', 'auto-imunes', 'banho-maria', 'batata-doce',
                                 'batatas-doce', 'bate-boca', 'bate-papo', 'bate-papos', 'beija-flor', 'beija-flores',
                                 'bem-estar', 'bem-sucedida', 'bem-sucedidas', 'bem-sucedido', 'bem-sucedidos',
                                 'bem-vinda', 'bem-vindas', 'bem-vindo', 'bem-vindos', 'boa-f??', 'boas-vindas',
                                 'boca-de-urna', 'caixa-preta', 'caixa-pretas', 'caixas-preta', 'caixas-pretas',
                                 'cana-de-a????car', 'canas-de-a????car', 'capim-elefante', 'carro-chefe', 'carro-forte',
                                 'carros-chefe', 'carros-forte', 'case-control', 'cd-rom', 'cd-roms', 'censo-designada',
                                 'censo-designadas', 'censo-designado', 'censo-designados', 'coca-cola', 'coca-colas',
                                 'contra-ataque', 'contra-ataques', 'cor-de-rosa', 'corpo-de-prova', 'corpo-de-provas',
                                 'corpos-de-provas', 'cross-sectional', 'curta-metragem', 'curta-metragens',
                                 'curto-circuito', 'curto-circuitos', 'c??lula-tronco', 'c??lulas-tronco', 'data-base',
                                 'data-bases', 'decreto-lei', 'decreto-leis', 'decretos-lei', 'decretos-leis',
                                 'desvio-padr??o', 'desvio-padr??es', 'desvios-padr??o', 'dia-a-dia', 'dona-de-casa',
                                 'donas-de-casa', 'double-blind', 'duplo-cego', 'e-mail', 'e-mails', 'ene-??sima',
                                 'ene-??simas', 'erva-mate', 'ervas-mate', 'follow-up', 'free-lance', 'free-lancers',
                                 'free-lances', 'gr??-bretanha', 'guarda-costas', 'guarda-roupa', 'guarda-roupas',
                                 'high-risk', 'hip-hop', 'i-??sima', 'i-??simas', 'infra-estrutura', 'infra-estruturas',
                                 'inter-relacionada', 'inter-relacionadas', 'inter-relacionado', 'inter-relacionados',
                                 'inter-relacionamentos', 'inter-rela????es', 'key-word', 'key-words', 'know-how',
                                 'long-term', 'longa-metragem', 'longa-metragens', 'lua-de-mel', 'luas-de-mel',
                                 'lugar-comum', 'l??ngua-alvo', 'l??nguas-alvo', 'mais-valia', 'mal-entendido',
                                 'mal-estar', 'mata-mata', 'mat??ria-prima', 'mat??ria-primas', 'mau-estar',
                                 'meia-atacante', 'meia-atacantes', 'meia-idade', 'meia-noite', 'meia-vida',
                                 'meia-vidas', 'meio-ambiente', 'meio-campo', 'meio-campos', 'meio-dia', 'meio-termo',
                                 'meio-termos', 'meios-campo', 'meios-termos', 'mercedes-benz', 'meta-analysis',
                                 'meu-entendido', 'm??-f??', 'm??o-de-obra', 'n-??sima', 'n-??simas', 'obra-prima',
                                 'obras-primas', 'oitava-de-final', 'oitavas-de-final', 'on-line', 'p-valor',
                                 'padr??o-ouro', 'padr??es-ouro', 'palavra-chave', 'palavras-chave', 'palavras-chaves',
                                 'pan-americana', 'pan-americanas', 'pan-americanos', 'pau-brasil', 'paus-brasil',
                                 'paus-brasis', 'pay-per-view', 'pay-per-views', 'peso-pesado', 'pesos-pesados',
                                 'pesquisa-a????o', 'pesquisas-a????o', 'placebo-controlled', 'population-based',
                                 'popula????o-alvo', 'popula????es-alvo', 'porta-avi??es', 'porta-enxerto', 'porta-enxertos',
                                 'porta-malas', 'porta-voz', 'porta-vozes', 'post-mortem', 'primeira-dama',
                                 'primeira-damas', 'primeiras-damas', 'pronto-atendimento', 'pronto-atendimentos',
                                 'pronto-socorro', 'pronto-socorros', 'prontos-atendimentos', 'prontos-socorros',
                                 'pr??-frontal', 'pr??-natais', 'pr??-natal', 'pr??-requisito', 'pr??-requisitos',
                                 'p??ra-quedas', 'p??r-do-sol', 'quadro-negro', 'quadros-negros', 'quarta-de-final',
                                 'quarta-feira', 'quartas-de-final', 'quartas-feiras', 'quarteis-generais',
                                 'quartel-general', 'quebra-cabe??a', 'quebra-cabe??as', 'qui-quadrado', 'quinta-feira',
                                 'quintas-feiras', 'raio-x', 'raios-x', 'r??-tim-bum', 'salva-vidas',
                                 'secret??rio-executivo', 'secret??rios-executivos', 'segunda-feira', 'segundas-feiras',
                                 'sexta-feira', 'sexta-feiras', 'sextas-feiras', 'short-term', 'springer-verlag',
                                 'super-homem', 'ter??a-feira', 'ter??as-feiras', 'todo-poderoso', 'ultra-som',
                                 'ultra-sonografia', 'ultra-sonografias', 'ultra-sonogr??fico', 'ultra-sonogr??ficos',
                                 'ultra-sons', 'ultra-s??nica', 'ultra-s??nico', 'ultra-s??nicos', 'ultra???s??nicas',
                                 'vice-versa', 'villa-lobos', 'vis-??-vis', 'x-man', 'x-men', 'x-ray', 'x-rays']
MULTI_WORD_CLASS_LIST = LEGAL_MULTIWORD_EXPRESSIONS + GENERAL_MULTIWORD_EXPRESSIONS
ROMAN_NUMERAL_REGEX = '^(?=[mdclxvi])m*(c[md]|d?c*)(x[cl]|l?x*)(i[xv]|v?i*)$'

PLACE_NAMES = ["s??o-paulo", "guarulhos", "campinas", "s??o-bernardo-do-campo",
               "s??o-jos??-dos-campos", "santo-andr??", "ribeir??o-preto", "osasco", "sorocaba",
               "mau??", "s??o-jos??-do-rio-preto", "mogi-das-cruzes", "santos", "diadema", "jundia??",
               "piracicaba", "carapicu??ba", "bauru", "itaquaquecetuba", "s??o-vicente", "franca",
               "praia-grande", "guaruj??", "taubat??", "limeira", "suzano", "tabo??o-da-serra",
               "sumar??", "barueri", "embu-das-artes", "s??o-carlos", "indaiatuba", "cotia",
               "americana", "mar??lia", "itapevi", "araraquara", "jacare??", "hortol??ndia",
               "presidente-prudente", "rio-claro", "ara??atuba", "ferraz-de-vasconcelos",
               "santa-b??rbara-d'oeste", "francisco-morato", "itapecerica-da-serra", "itu",
               "bragan??a-paulista", "pindamonhangaba", "itapetininga", "s??o-caetano-do-sul",
               "franco-da-rocha", "mogi-gua??u", "ja??", "botucatu", "atibaia",
               "santana-de-parna??ba", "araras", "cubat??o", "valinhos", "sert??ozinho", "jandira",
               "birigui", "ribeir??o-pires", "votorantim", "barretos", "catanduva",
               "v??rzea-paulista", "guaratinguet??", "tatu??", "caraguatatuba", "itatiba", "salto",
               "po??", "ourinhos", "paul??nia", "assis", "leme", "itanha??m", "caieiras",
               "mairipor??", "votuporanga", "itapeva", "ca??apava", "mogi-mirim",
               "s??o-jo??o-da-boa-vista", "s??o-roque", "ubatuba", "avar??", "aruj??", "s??o-sebasti??o",
               "lorena", "campo-limpo-paulista", "mat??o", "cruzeiro", "ibi??na", "vinhedo", "lins",
               "bebedouro", "jaboticabal", "cajamar", "pirassununga", "itapira", "cosm??polis",
               "amparo", "embu-gua??u", "fernand??polis", "mococa", "len????is-paulista", "peru??be",
               "tup??", "pen??polis", "bertioga", "batatais", "itupeva", "boituva", "nova-odessa",
               "ibitinga", "mirassol", "monte-mor", "jaguari??na", "santa-isabel", "taquaritinga",
               "andradina", "mongagu??", "registro", "porto-ferreira", "capivari", "piedade",
               "s??o-jos??-do-rio-pardo", "ol??mpia", "artur-nogueira", "porto-feliz",
               "vargem-grande-paulista", "campos-do-jord??o", "s??o-joaquim-da-barra",
               "rio-grande-da-serra", "itarar??", "monte-alto", "pontal", "cabre??va", "jales",
               "cerquilho", "louveira", "pedreira", "santa-cruz-do-rio-pardo", "trememb??",
               "mairinque", "cap??o-bonito", "dracena", "pederneiras", "paragua??u-paulista",
               "salto-de-pirapora", "serrana", "gar??a", "jardin??polis",
               "esp??rito-santo-do-pinhal", "presidente-epit??cio", "orl??ndia", "it??polis",
               "vargem-grande-do-sul", "tiet??", "ituverava", "novo-horizonte", "socorro",
               "s??o-manuel", "gua??ra", "am??rico-brasiliense", "promiss??o", "guariba",
               "pitangueiras", "presidente-venceslau", "agudos", "iper??", "jos??-bonif??cio",
               "agua??", "aparecida", "barra-bonita", "s??o-pedro", "cravinhos", "bariri",
               "rio-das-pedras", "ibat??", "adamantina", "ilhabela", "santa-cruz-das-palmeiras",
               "ara??oiaba-da-serra", "descalvado", "cachoeira-paulista", "morro-agudo",
               "guararapes", "s??o-miguel-arcanjo", "osvaldo-cruz", "barrinha", "biritiba-mirim",
               "santa-f??-do-sul", "juquitiba", "c??ndido-mota", "iguape", "igarapava",
               "casa-branca", "jarinu", "piraju", "guararema", "rancharia", "mirand??polis",
               "serra-negra", "pilar-do-sul", "cajati", "laranjal-paulista", "conchal",
               "santa-rita-do-passa-quatro", "pirapozinho", "dois-c??rregos", "piracaia", "ita??",
               "santa-gertrudes", "ilha-solteira", "santa-rosa-de-viterbo", "valpara??so",
               "martin??polis", "cajuru", "tanabi", "pereira-barreto", "piraju??",
               "bom-jesus-dos-perd??es", "angatuba", "monte-apraz??vel", "brodowski",
               "??lvares-machado", "igara??u-do-tiet??", "potim", "cordeir??polis", "brotas", "apia??",
               "iracem??polis", "santo-ant??nio-de-posse", "taquarituba", "tamba??",
               "teodoro-sampaio", "ara??ariguama", "miguel??polis", "palmital", "pompeia",
               "luc??lia", "cunha", "nova-granada", "prad??polis", "guapia??u", "guar??", "castilho",
               "bastos", "santo-anast??cio", "engenheiro-coelho", "capela-do-alto", "itatinga",
               "junqueir??polis", "regente-feij??", "paranapanema", "cerqueira-c??sar", "buri",
               "miracatu", "pariquera-a??u", "monte-azul-paulista", "caconde", "viradouro",
               "pirapora-do-bom-jesus", "juqui??", "??guas-de-lindoia", "alum??nio",
               "nazar??-paulista", "guare??", "colina", "mirante-do-paranapanema", "paraibuna",
               "itirapina", "ces??rio-lange", "conchas", "jacupiranga", "elias-fausto",
               "cafel??ndia", "itaber??", "bady-bassitt", "sever??nia", "itariri", "potirendaba",
               "charqueada", "macatuba", "guapiara", "buritama", "sales??polis", "pindorama",
               "pedregulho", "rosana", "tabatinga", "ribeir??o-branco", "ipu??", "altin??polis",
               "borborema", "fartura", "s??o-louren??o-da-serra", "panorama", "tupi-paulista",
               "eldorado", "santa-ad??lia", "s??o-sim??o", "itajobi", "pinhalzinho", "auriflama",
               "itaporanga", "tarum??", "ipaussu", "luiz-ant??nio", "holambra",
               "boa-esperan??a-do-sul", "santa-branca", "patroc??nio-paulista", "serra-azul",
               "fl??rida-paulista", "pacaembu", "quat??", "maraca??", "itapu??", "tagua??", "urup??s",
               "piquete", "avanhandava", "piratininga", "morungaba", "queluz", "valentim-gentil",
               "palmares-paulista", "joan??polis", "ribeir??o-bonito", "presidente-bernardes",
               "tapiratiba", "palestina", "mineiros-do-tiet??", "sete-barras", "cananeia",
               "riol??ndia", "duartina", "chavantes", "tabapu??", "ibir??", "bocaina", "cardoso",
               "s??o-sebasti??o-da-grama", "guai??ara", "lav??nia", "sales-oliveira", "bofete",
               "iacanga", "nhandeara", "pirangi", "getulina", "pedro-de-toledo", "estiva-gerbi",
               "guaraci", "nova-europa", "ilha-comprida", "bernardino-de-campos", "divinol??ndia",
               "arei??polis", "parapu??", "bananal", "s??o-bento-do-sapuca??", "general-salgado",
               "vera-cruz", "rinc??o", "roseira", "s??o-luiz-do-paraitinga", "cajobi", "vargem",
               "ouroeste", "sarapu??", "uchoa", "torrinha", "rin??polis", "porangaba", "dumont",
               "manduri", "nova-campina", "ariranha", "regin??polis", "igarat??", "hercul??ndia",
               "euclides-da-cunha-paulista", "terra-roxa", "salto-grande", "palmeira-d'oeste",
               "iaras", "cedral", "ur??nia", "rafard", "b??lsamo", "paulo-de-faria",
               "neves-paulista", "dobrada", "s??o-jos??-da-bela-vista", "dourado", "santa-l??cia",
               "vista-alegre-do-alto", "pereiras", "cristais-paulista", "clementina",
               "ouro-verde", "arealva", "santo-ant??nio-do-aracangu??", "estrela-d'oeste",
               "guara??a??", "irapuru", "saltinho", "ic??m", "??guas-da-prata", "iep??", "macaubal",
               "monte-alegre-do-sul", "bilac", "irapu??", "lindoia", "itobi", "tapira??", "catigu??",
               "ibirarema", "sud-mennucci", "ribeir??o-grande", "s??o-pedro-do-turvo",
               "barra-do-turvo", "guatapar??", "restinga", "ipe??na", "tarabai", "nuporanga",
               "barbosa", "pauliceia", "cosmorama", "lavrinhas", "jaci", "orindi??va",
               "nova-alian??a", "jaborandi", "santo-ant??nio-da-alegria", "tuiuti",
               "santo-ant??nio-do-pinhal", "anhembi", "guarant??", "natividade-da-serra",
               "quintana", "jambeiro", "g??lia", "oriente", "itirapu??", "para??so", "pardinho",
               "arandu", "sales", "iacri", "ubarana", "silveiras", "taia??u", "taciba", "col??mbia",
               "santa-maria-da-serra", "echapor??", "pedra-bela", "??guas-de-santa-b??rbara",
               "poloni", "coroados", "alambari", "campina-do-monte-alegre", "santa-albertina",
               "piacatu", "am??rico-de-campos", "santo-ant??nio-do-jardim", "caiu??",
               "marab??-paulista", "taquariva??", "novais", "tr??s-fronteiras", "luizi??nia",
               "fernando-prestes", "guaimb??", "balbinos", "barra-do-chap??u", "bra??na", "aramina",
               "santa-ernestina", "sabino", "tai??va", "riversul", "mendon??a", "ava??", "ipigu??",
               "salmour??o", "guzol??ndia", "prat??nia", "planalto", "??lvaro-de-carvalho", "canitar",
               "nipo??", "canas", "anal??ndia", "campos-novos-paulista", "itapura", "lagoinha",
               "indiana", "mirassol??ndia", "narandiba", "esp??rito-santo-do-turvo", "boraceia",
               "glic??rio", "gast??o-vidigal", "gavi??o-peixoto", "ubirajara",
               "sant??polis-do-aguape??", "j??lio-mesquita", "motuca", "ribeir??o-corrente",
               "coronel-macedo", "monteiro-lobato", "pontalinda", "lup??rcio", "ribeir??o-do-sul",
               "tejup??", "jo??o-ramalho", "santa-cruz-da-concei????o", "murutinga-do-sul",
               "buritizal", "onda-verde", "sandovalina", "ocau??u", "cabr??lia-paulista",
               "itapirapu??-paulista", "iporanga", "aparecida-d'oeste", "caiabu", "populina",
               "alfredo-marcondes", "monte-castelo", "altair", "s??o-jos??-do-barreiro", "anhumas",
               "nova-luzit??nia", "alto-alegre", "presidente-alves", "pracinha", "mari??polis",
               "paranapu??", "corumbata??", "in??bia-paulista", "nova-independ??ncia", "suzan??polis",
               "bom-sucesso-de-itarar??", "indiapor??", "areias", "reden????o-da-serra", "meridiano",
               "itaju", "quadra", "maced??nia", "piquerobi", "??lvares-florence", "elisi??rio",
               "sarutai??", "rifaina", "adolfo", "platina", "sebastian??polis-do-sul", "mombuca",
               "bar??o-de-antonina", "??guas-de-s??o-pedro", "ponga??", "queiroz", "jumirim",
               "ribeira", "itaoca", "alvinl??ndia", "emilian??polis", "jeriquara", "rubineia",
               "nantes", "rubi??cea", "magda", "santo-expedito", "pedrinhas-paulista",
               "mira-estrela", "marapoama", "bento-de-abreu", "santa-mercedes", "floreal",
               "brejo-alegre", "s??o-francisco", "taquaral", "c??ndido-rodrigues",
               "gabriel-monteiro", "estrela-do-norte", "zacarias", "flor??nea", "timburi",
               "borebi", "lut??cia", "oscar-bressane", "pontes-gestal", "s??o-jo??o-das-duas-pontes",
               "c??ssia-dos-coqueiros", "santa-rita-d'oeste", "??leo", "pedran??polis", "arape??",
               "emba??ba", "sagres", "torre-de-pedra", "lucian??polis", "nova-guataporanga",
               "lourdes", "mon????es", "ribeir??o-dos-??ndios", "parisi", "santa-cruz-da-esperan??a",
               "dolcin??polis", "santa-clara-d'oeste", "marin??polis", "s??o-jo??o-do-pau-d'alho",
               "cruz??lia", "turi??ba", "guarani-d'oeste", "s??o-jo??o-de-iracema", "mes??polis",
               "nova-cana??-paulista", "uni??o-paulista", "vit??ria-brasil", "paulist??nia",
               "asp??sia", "dirce-reis", "arco-??ris", "turmalina", "trabiju", "fern??o",
               "santa-salete", "santana-da-ponte-pensa", "flora-rica", "nova-castilho", "uru",
               "bor??", "s??o_paulo", "guarulhos", "campinas", "s??o_bernardo_do_campo",
               "s??o_jos??_dos_campos", "santo_andr??", "ribeir??o_preto", "osasco", "sorocaba",
               "mau??", "s??o_jos??_do_rio_preto", "mogi_das_cruzes", "santos", "diadema", "jundia??",
               "piracicaba", "carapicu??ba", "bauru", "itaquaquecetuba", "s??o_vicente", "franca",
               "praia_grande", "guaruj??", "taubat??", "limeira", "suzano", "tabo??o_da_serra",
               "sumar??", "barueri", "embu_das_artes", "s??o_carlos", "indaiatuba", "cotia",
               "americana", "mar??lia", "itapevi", "araraquara", "jacare??", "hortol??ndia",
               "presidente_prudente", "rio_claro", "ara??atuba", "ferraz_de_vasconcelos",
               "santa_b??rbara_d'oeste", "francisco_morato", "itapecerica_da_serra", "itu",
               "bragan??a_paulista", "pindamonhangaba", "itapetininga", "s??o_caetano_do_sul",
               "franco_da_rocha", "mogi_gua??u", "ja??", "botucatu", "atibaia",
               "santana_de_parna??ba", "araras", "cubat??o", "valinhos", "sert??ozinho", "jandira",
               "birigui", "ribeir??o_pires", "votorantim", "barretos", "catanduva",
               "v??rzea_paulista", "guaratinguet??", "tatu??", "caraguatatuba", "itatiba", "salto",
               "po??", "ourinhos", "paul??nia", "assis", "leme", "itanha??m", "caieiras",
               "mairipor??", "votuporanga", "itapeva", "ca??apava", "mogi_mirim",
               "s??o_jo??o_da_boa_vista", "s??o_roque", "ubatuba", "avar??", "aruj??", "s??o_sebasti??o",
               "lorena", "campo_limpo_paulista", "mat??o", "cruzeiro", "ibi??na", "vinhedo", "lins",
               "bebedouro", "jaboticabal", "cajamar", "pirassununga", "itapira", "cosm??polis",
               "amparo", "embu-gua??u", "fernand??polis", "mococa", "len????is_paulista", "peru??be",
               "tup??", "pen??polis", "bertioga", "batatais", "itupeva", "boituva", "nova_odessa",
               "ibitinga", "mirassol", "monte_mor", "jaguari??na", "santa_isabel", "taquaritinga",
               "andradina", "mongagu??", "registro", "porto_ferreira", "capivari", "piedade",
               "s??o_jos??_do_rio_pardo", "ol??mpia", "artur_nogueira", "porto_feliz",
               "vargem_grande_paulista", "campos_do_jord??o", "s??o_joaquim_da_barra",
               "rio_grande_da_serra", "itarar??", "monte_alto", "pontal", "cabre??va", "jales",
               "cerquilho", "louveira", "pedreira", "santa_cruz_do_rio_pardo", "trememb??",
               "mairinque", "cap??o_bonito", "dracena", "pederneiras", "paragua??u_paulista",
               "salto_de_pirapora", "serrana", "gar??a", "jardin??polis",
               "esp??rito_santo_do_pinhal", "presidente_epit??cio", "orl??ndia", "it??polis",
               "vargem_grande_do_sul", "tiet??", "ituverava", "novo_horizonte", "socorro",
               "s??o_manuel", "gua??ra", "am??rico_brasiliense", "promiss??o", "guariba",
               "pitangueiras", "presidente_venceslau", "agudos", "iper??", "jos??_bonif??cio",
               "agua??", "aparecida", "barra_bonita", "s??o_pedro", "cravinhos", "bariri",
               "rio_das_pedras", "ibat??", "adamantina", "ilhabela", "santa_cruz_das_palmeiras",
               "ara??oiaba_da_serra", "descalvado", "cachoeira_paulista", "morro_agudo",
               "guararapes", "s??o_miguel_arcanjo", "osvaldo_cruz", "barrinha", "biritiba_mirim",
               "santa_f??_do_sul", "juquitiba", "c??ndido_mota", "iguape", "igarapava",
               "casa_branca", "jarinu", "piraju", "guararema", "rancharia", "mirand??polis",
               "serra_negra", "pilar_do_sul", "cajati", "laranjal_paulista", "conchal",
               "santa_rita_do_passa_quatro", "pirapozinho", "dois_c??rregos", "piracaia", "ita??",
               "santa_gertrudes", "ilha_solteira", "santa_rosa_de_viterbo", "valpara??so",
               "martin??polis", "cajuru", "tanabi", "pereira_barreto", "piraju??",
               "bom_jesus_dos_perd??es", "angatuba", "monte_apraz??vel", "brodowski",
               "??lvares_machado", "igara??u_do_tiet??", "potim", "cordeir??polis", "brotas", "apia??",
               "iracem??polis", "santo_ant??nio_de_posse", "taquarituba", "tamba??",
               "teodoro_sampaio", "ara??ariguama", "miguel??polis", "palmital", "pompeia",
               "luc??lia", "cunha", "nova_granada", "prad??polis", "guapia??u", "guar??", "castilho",
               "bastos", "santo_anast??cio", "engenheiro_coelho", "capela_do_alto", "itatinga",
               "junqueir??polis", "regente_feij??", "paranapanema", "cerqueira_c??sar", "buri",
               "miracatu", "pariquera-a??u", "monte_azul_paulista", "caconde", "viradouro",
               "pirapora_do_bom_jesus", "juqui??", "??guas_de_lindoia", "alum??nio",
               "nazar??_paulista", "guare??", "colina", "mirante_do_paranapanema", "paraibuna",
               "itirapina", "ces??rio_lange", "conchas", "jacupiranga", "elias_fausto",
               "cafel??ndia", "itaber??", "bady_bassitt", "sever??nia", "itariri", "potirendaba",
               "charqueada", "macatuba", "guapiara", "buritama", "sales??polis", "pindorama",
               "pedregulho", "rosana", "tabatinga", "ribeir??o_branco", "ipu??", "altin??polis",
               "borborema", "fartura", "s??o_louren??o_da_serra", "panorama", "tupi_paulista",
               "eldorado", "santa_ad??lia", "s??o_sim??o", "itajobi", "pinhalzinho", "auriflama",
               "itaporanga", "tarum??", "ipaussu", "luiz_ant??nio", "holambra",
               "boa_esperan??a_do_sul", "santa_branca", "patroc??nio_paulista", "serra_azul",
               "fl??rida_paulista", "pacaembu", "quat??", "maraca??", "itapu??", "tagua??", "urup??s",
               "piquete", "avanhandava", "piratininga", "morungaba", "queluz", "valentim_gentil",
               "palmares_paulista", "joan??polis", "ribeir??o_bonito", "presidente_bernardes",
               "tapiratiba", "palestina", "mineiros_do_tiet??", "sete_barras", "cananeia",
               "riol??ndia", "duartina", "chavantes", "tabapu??", "ibir??", "bocaina", "cardoso",
               "s??o_sebasti??o_da_grama", "guai??ara", "lav??nia", "sales_oliveira", "bofete",
               "iacanga", "nhandeara", "pirangi", "getulina", "pedro_de_toledo", "estiva_gerbi",
               "guaraci", "nova_europa", "ilha_comprida", "bernardino_de_campos", "divinol??ndia",
               "arei??polis", "parapu??", "bananal", "s??o_bento_do_sapuca??", "general_salgado",
               "vera_cruz", "rinc??o", "roseira", "s??o_luiz_do_paraitinga", "cajobi", "vargem",
               "ouroeste", "sarapu??", "uchoa", "torrinha", "rin??polis", "porangaba", "dumont",
               "manduri", "nova_campina", "ariranha", "regin??polis", "igarat??", "hercul??ndia",
               "euclides_da_cunha_paulista", "terra_roxa", "salto_grande", "palmeira_d'oeste",
               "iaras", "cedral", "ur??nia", "rafard", "b??lsamo", "paulo_de_faria",
               "neves_paulista", "dobrada", "s??o_jos??_da_bela_vista", "dourado", "santa_l??cia",
               "vista_alegre_do_alto", "pereiras", "cristais_paulista", "clementina",
               "ouro_verde", "arealva", "santo_ant??nio_do_aracangu??", "estrela_d'oeste",
               "guara??a??", "irapuru", "saltinho", "ic??m", "??guas_da_prata", "iep??", "macaubal",
               "monte_alegre_do_sul", "bilac", "irapu??", "lindoia", "itobi", "tapira??", "catigu??",
               "ibirarema", "sud_mennucci", "ribeir??o_grande", "s??o_pedro_do_turvo",
               "barra_do_turvo", "guatapar??", "restinga", "ipe??na", "tarabai", "nuporanga",
               "barbosa", "pauliceia", "cosmorama", "lavrinhas", "jaci", "orindi??va",
               "nova_alian??a", "jaborandi", "santo_ant??nio_da_alegria", "tuiuti",
               "santo_ant??nio_do_pinhal", "anhembi", "guarant??", "natividade_da_serra",
               "quintana", "jambeiro", "g??lia", "oriente", "itirapu??", "para??so", "pardinho",
               "arandu", "sales", "iacri", "ubarana", "silveiras", "taia??u", "taciba", "col??mbia",
               "santa_maria_da_serra", "echapor??", "pedra_bela", "??guas_de_santa_b??rbara",
               "poloni", "coroados", "alambari", "campina_do_monte_alegre", "santa_albertina",
               "piacatu", "am??rico_de_campos", "santo_ant??nio_do_jardim", "caiu??",
               "marab??_paulista", "taquariva??", "novais", "tr??s_fronteiras", "luizi??nia",
               "fernando_prestes", "guaimb??", "balbinos", "barra_do_chap??u", "bra??na", "aramina",
               "santa_ernestina", "sabino", "tai??va", "riversul", "mendon??a", "ava??", "ipigu??",
               "salmour??o", "guzol??ndia", "prat??nia", "planalto", "??lvaro_de_carvalho", "canitar",
               "nipo??", "canas", "anal??ndia", "campos_novos_paulista", "itapura", "lagoinha",
               "indiana", "mirassol??ndia", "narandiba", "esp??rito_santo_do_turvo", "boraceia",
               "glic??rio", "gast??o_vidigal", "gavi??o_peixoto", "ubirajara",
               "sant??polis_do_aguape??", "j??lio_mesquita", "motuca", "ribeir??o_corrente",
               "coronel_macedo", "monteiro_lobato", "pontalinda", "lup??rcio", "ribeir??o_do_sul",
               "tejup??", "jo??o_ramalho", "santa_cruz_da_concei????o", "murutinga_do_sul",
               "buritizal", "onda_verde", "sandovalina", "ocau??u", "cabr??lia_paulista",
               "itapirapu??_paulista", "iporanga", "aparecida_d'oeste", "caiabu", "populina",
               "alfredo_marcondes", "monte_castelo", "altair", "s??o_jos??_do_barreiro", "anhumas",
               "nova_luzit??nia", "alto_alegre", "presidente_alves", "pracinha", "mari??polis",
               "paranapu??", "corumbata??", "in??bia_paulista", "nova_independ??ncia", "suzan??polis",
               "bom_sucesso_de_itarar??", "indiapor??", "areias", "reden????o_da_serra", "meridiano",
               "itaju", "quadra", "maced??nia", "piquerobi", "??lvares_florence", "elisi??rio",
               "sarutai??", "rifaina", "adolfo", "platina", "sebastian??polis_do_sul", "mombuca",
               "bar??o_de_antonina", "??guas_de_s??o_pedro", "ponga??", "queiroz", "jumirim",
               "ribeira", "itaoca", "alvinl??ndia", "emilian??polis", "jeriquara", "rubineia",
               "nantes", "rubi??cea", "magda", "santo_expedito", "pedrinhas_paulista",
               "mira_estrela", "marapoama", "bento_de_abreu", "santa_mercedes", "floreal",
               "brejo_alegre", "s??o_francisco", "taquaral", "c??ndido_rodrigues",
               "gabriel_monteiro", "estrela_do_norte", "zacarias", "flor??nea", "timburi",
               "borebi", "lut??cia", "oscar_bressane", "pontes_gestal", "s??o_jo??o_das_duas_pontes",
               "c??ssia_dos_coqueiros", "santa_rita_d'oeste", "??leo", "pedran??polis", "arape??",
               "emba??ba", "sagres", "torre_de_pedra", "lucian??polis", "nova_guataporanga",
               "lourdes", "mon????es", "ribeir??o_dos_??ndios", "parisi", "santa_cruz_da_esperan??a",
               "dolcin??polis", "santa_clara_d'oeste", "marin??polis", "s??o_jo??o_do_pau_d'alho",
               "cruz??lia", "turi??ba", "guarani_d'oeste", "s??o_jo??o_de_iracema", "mes??polis",
               "nova_cana??_paulista", "uni??o_paulista", "vit??ria_brasil", "paulist??nia",
               "asp??sia", "dirce_reis", "arco-??ris", "turmalina", "trabiju", "fern??o",
               "santa_salete", "santana_da_ponte_pensa", "flora_rica", "nova_castilho", "uru",
               "bor??", "acre", "alagoas", "amap??", "amazonas", "bahia", "cear??",
               "distrito_federal", "esp??rito_santo", "goi??s", "maranh??o", "mato_grosso",
               "mato_grosso_do_sul", "minas_gerais", "par??", "para??ba", "paran??", "pernambuco",
               "piau??", "rio_de_janeiro", "rio_grande_do_norte", "rio_grande_do_sul", "rond??nia",
               "roraima", "santa_catarina", "s??o_paulo", "sergipe", "tocantins", "rio_branco",
               "macei??", "macap??", "manaus", "salvador", "fortaleza", "bras??lia", "vit??ria",
               "goi??nia", "s??o_lu??s", "cuiab??", "campo_grande", "belo_horizonte", "bel??m",
               "jo??o_pessoa", "curitiba", "recife", "teresina", "rio_de_janeiro", "natal",
               "porto_alegre", "porto_velho", "boa_vista", "florian??polis", "s??o_paulo",
               "aracaju", "palmas", "acre", "alagoas", "amap??", "amazonas", "bahia", "cear??",
               "distrito-federal", "esp??rito-santo", "goi??s", "maranh??o", "mato-grosso",
               "mato-grosso-do-sul", "minas-gerais", "par??", "para??ba", "paran??", "pernambuco",
               "piau??", "rio-de-janeiro", "rio-grande-do-norte", "rio-grande-do-sul", "rond??nia",
               "roraima", "santa-catarina", "s??o-paulo", "sergipe", "tocantins", "rio-branco",
               "macei??", "macap??", "manaus", "salvador", "fortaleza", "bras??lia", "vit??ria",
               "goi??nia", "s??o-lu??s", "cuiab??", "campo-grande", "belo-horizonte", "bel??m",
               "jo??o-pessoa", "curitiba", "recife", "teresina", "rio-de-janeiro", "natal",
               "porto-alegre", "porto-velho", "boa-vista", "florian??polis", "s??o-paulo",
               "aracaju", "palmas"]
TYPOS = ['', '"', 'eo', 'ea', 'eos', 'au', 'eas', 'ede', 'eda', 'edo', 'edu', 'edocumentos,', 'acerca', 'ad', 'cara',
         'car', 'f', 'g', 'h', 'in', 'j', 'n', 's', 's/a']
SELECTED_PROPER_NAMES = ['ad??lia', 'andr??', 'b??rbara', 'barbara', 'barreto', 'bento', 'bernardo', 'bragan??a', 'caetano',
                         'c??ndido', 'carlos', 'carvalho', 'cassia', 'catarina', 'concei????o', 'dirce', 'embu', 'gua??u',
                         'gua????', 'isabel', 'itapecerica', 'itaquera', 'jo??o', 'joaquim', 'jord??o', 'maria', 'mateus',
                         'mogi', 'moji', 'monteiro', 'mo??ca', 'mooca', 'moreira', 'nogueira', 'palmares', 'paragua??u',
                         'paulo', 'paulinia', 'padrinhas', 'pedro', 'penha', 'rita', 'roque', 'sapuca??', 'sebasti??o',
                         'sim??o', 'vicente']
STOPWORDS = nltk.corpus.stopwords.words(
    'portuguese') + PLACE_NAMES + TYPOS + SELECTED_PROPER_NAMES


TO_REMOVE_FIRST_CHAR = ['eobjetiva', 'eemolumentos', 'eanexos', '??execu????o', 'aadministra????o', 'eimpenhorabilidade',
                        'eambiental', 'aexecu????o', 'eincomunicabilidade', 'eautentica????o', 'eapreens??o', 'eauxiliares',
                        'eespecifica????o', 'aalienacao', 'e??bito', 'aoutorga', 'evenda', 'edocumentos', 'etabeli??o',
                        'eregistrais', 'ecompra', 'equalitativa', 'equantitativa', 'et??tulos', 'etabelionato',
                        'eperpetuidade', 'etribut??rio', 'econtribui????es', 'eregistros', 'eratifica????o', 'eregistral',
                        'ec??lculo', 'ereceita', 'eregistro', 'enotarial', 'econfronta????o', 'edeterminada', 'esubjetiva',
                        'enotas', 'etransfer??ncia', 'econfronta????es', 'ep??', 'etutelas', 'ecivil', 'eregistradores',
                        'ecpf', 'eprotesto', 'eprocura????es', 'epr??tica', 'econven????o', 'en??o', 'efiscal', 'ehipotec??ria',
                        'ecess??o', 'epartilha', 'edistribui????o', 'edespesa', 'epessoais', 'ecorrei????es', 'efinanceira',
                        'epra??a', 'apreliminar', 'opedido', 'oper??odo', 'aterceiro', 'areda????o', 'aresolu????o', 'alei',
                        '??lavratura', '??aliena????o', 'aaliena????o', 'acontagem', '??delega????o', 'adelega????o', 'adelegados',
                        '??lei', 'amaior', '??maior', 'aregistro', '??registro', 'aterceiros', '??terceiro', '??terceiros',
                        'e??rea', 'earea', 'e??reas', 'ecertid??o', 'ecertid??es', 'ecerto', 'et??tulo', 'etitulos', '??titular',
                        'eurban??stico', 'eurban??stica']
TO_REMOVE_LAST_CHAR = ['im??veis,', 'paulista,', 'ad_mensuram"']

SPECIAL_STEM_DICT = {'ac??rd??o': 'acord__', 'ac??rd??os': 'acord__', 'anal??gica': 'analog__', 'apelido': 'apel__',
                     'assento': 'assent__', 'assentos': 'assent__', '??udio': 'audi__', 'autor': 'autor__',
                     'autoria': 'autor__', 'banc??rio': 'banc__', 'carta': 'cart__', 'cartas': 'cart__',
                     'carteira': 'cart___', 'casamento': 'cas__', 'casada': 'cas__', 'casamentos': 'cas__',
                     'casado': 'cas__', 'casados': 'cas__', 'caso': 'cas___', 'casos': 'cas___', 'certid??o': 'cert__',
                     'certid??es': 'cert__', 'cessa????o': 'cess__', 'cessar': 'cess__', 'cessada': 'cess__',
                     'cessantes': 'cess__', 'cidade': 'cidad__', 'comprida': 'compr__', 'consultivo': 'consul__',
                     'consumado': 'consum__', 'contado': 'cont', 'contra????o': 'contr', 'contratantes': 'contrat__',
                     'contrata????o': 'contrat__', 'contratado': 'contrat__', 'correi????o': 'corre__',
                     'correi????es': 'corre__', 'corretivos': 'corre__', 'corregedor': 'correg__',
                     'corregedores': 'correg__', 'diretoria': 'diret__', 'd??lar': 'dol__', 'equipe': 'equip__',
                     'estri????o': 'estr__', 'extra????o': 'extr__', 'fila': 'fil__', 'funda': 'fund__', 'fundo': 'fund__',
                     'garagem': 'gar__', 'general': 'gener__', 'ger??ncia': 'ger__', 'ilegal': 'ileg__',
                     'imposta': 'impost__', 'incri????o': 'incr__', 'informalismo': 'inform__', 'informal': 'inform__',
                     'inter': 'int__', '??ntimo': 'intim__', 'intimidade': 'intim__', 'jornadas': 'jorn__',
                     'jornada': 'jorn__', 'juntada': 'junt__', 'juros': 'jur', 'liquidez': 'liquid__',
                     'liquidante': 'liquid__', 'livro': 'livr__', 'livros': 'livr__', 'maio': 'mai__',
                     'mandat??rio': 'mandat__', 'mar??timas': 'marit__', 'minas': 'min__', 'muta????o': 'mut__',
                     'palmo': 'palm', 'parcial': 'parc__', 'parto': 'part__', 'partos': 'part__', 'plant??o': 'plant__',
                     'pontos': 'pont__', 'ponto': 'pont__', 'portaria': 'port__', 'portarias': 'port__',
                     'portal': 'port__', 'porto': 'port___', 'posturas': 'post__', 'prec??rio': 'prec__',
                     'preta????es': 'pret__', 'procedimento': 'proced__', 'procedimentos': 'proced__', 'provas': 'prov__',
                     'prova': 'prov__', 'prov': 'prov__', 'p??blico': 'public__', 'p??blica': 'public__',
                     'p??blicas': 'public__', 'p??blicos': 'public__', 'redondo': 'red__', 'rede': 'red__',
                     'regi??o': 'reg__', 'regulamento': 'regul__', 'relato': 'relat__', 'relevo': 'relev__',
                     'residual': 'resid__', 's??bados': 'sab__', 'sete': 'set__', 'sindic??ncia': 'sindic__',
                     'sindic??ncias': 'sindic__', 's??cia': 'soc__', 'solo': 'sol__', 'solteira': 'solt__',
                     'solteiro': 'solt__', 'tabela': 'tabel__', 'tabelas': 'tabel__', 'terra??o': 'terr__',
                     'testador': 'test__', 'teste': 'test__', 'universidade': 'univers__', 'vais': 'val__',
                     'v??lida': 'val__', 'v??lido': 'val__', 'var??o': 'var__', 'verba': 'verb__', 'verbas': 'verb__',
                     'vidual': 'vid__', 'vig??rio': 'vig__', 'vigor': 'vig___', 'vila': 'vil__', 'acesso': 'acess__'}

SPECIAL_STEM_COUNT_DICT = copy.deepcopy(SPECIAL_STEM_DICT)
for k in SPECIAL_STEM_COUNT_DICT:
    SPECIAL_STEM_COUNT_DICT[k] = 0

MERGE_DICT = {'abusiv': 'mc1', 'acess__': 'mc2', 'aca': 'mc3', 'advocac': 'mc4', 'agrar': 'mc5', 'alien': 'mc6',
              'ampl': 'mc7', 'anteri': 'mc8', 'ano': 'mc9', 'anul': 'mc10', 'apreens': 'mc11', 'aquis': 'mc12',
              'arpen': 'mc13', 'aus': 'mc14', 'autent': 'mc15', 'averbaca': 'mc16', 'banc__': 'mc17', 'boa-fe': 'mc18',
              'bonit': 'mc19', 'causal': 'mc20', 'cgj': 'mc21', 'civel': 'mc22', 'cnd': 'mc23', 'comerc': 'mc24',
              'compens': 'mc25', 'compet': 'mc26', 'cond': 'mc27', 'condomin': 'mc28', 'conjun': 'mc29',
              'consens': 'mc30', 'constitu': 'mc31', 'constru': 'mc32', 'consul__': 'mc33', 'contribu': 'mc34',
              'corre__': 'mc35', 'correg__': 'mc36', 'dan': 'mc37', 'decid': 'mc38', 'declar': 'mc39', 'demarc': 'mc40',
              'descr': 'mc41', 'destac': 'mc42', 'diret__': 'mc43', 'discrimin': 'mc44', 'doac': 'mc45',
              'domin': 'mc46', 'dupl': 'mc47', 'duv': 'mc48', 'ele': 'mc49', 'emiss': 'mc50', 'empres': 'mc51',
              'endoss': 'mc52', 'escrit': 'mc53', 'especi': 'mc54', 'especif': 'mc55', 'estatut': 'mc56',
              'exam': 'mc57', 'exced': 'mc58', 'excec': 'mc59', 'execuc': 'mc60', 'exig': 'mc61', 'expropri': 'mc62',
              'extinc': 'mc63', 'extraordin': 'mc64', 'fals': 'mc65', 'famil': 'mc66', 'faz': 'mc67', 'fiduci': 'mc68',
              'filh': 'mc69', 'fim': 'mc70', 'form': 'mc71', 'frac': 'mc72', 'fraud': 'mc73', 'gar': 'mc74',
              'ger__': 'mc75', 'gratu': 'mc76', 'habit': 'mc77', 'heranc': 'mc78', 'hipotec': 'mc79', 'ident': 'mc80',
              'imobili': 'mc81', 'imperfe': 'mc82', 'impost__': 'mc83', 'inadimpl': 'mc84', 'incr__': 'mc85',
              'indefer': 'mc86', 'indi': 'mc87', 'indispon': 'mc88', 'individ': 'mc89', 'indivis': 'mc90',
              'inexist': 'mc91', 'infra-estrut': 'mc92', 'inic': 'mc93', 'inscr': 'mc94', 'institu': 'mc95',
              'intempes': 'mc96', 'interpos': 'mc97', 'interven': 'mc98', 'invi': 'mc99', 'irregul': 'mc100',
              'judic': 'mc101', 'jurisd': 'mc102', 'just': 'mc103', 'lavr': 'mc104', 'legit': 'mc105', 'loc': 'mc106',
              'lot': 'mc107', 'ma-fe': 'mc108', 'magistr': 'mc109', 'mandat__': 'mc110', 'marit__': 'mc111',
              'matrimoni': 'mc112', 'mod': 'mc113', 'mor': 'mc114', 'municip': 'mc115', 'nacion': 'mc116',
              'necess': 'mc117', 'not': 'mc118', 'nul': 'mc119', 'obrig': 'mc120', 'ofic': 'mc121', 'ordin': 'mc122',
              'orig': 'mc123', 'padr': 'mc124', 'patrimon': 'mc125', 'pen': 'mc126', 'penh': 'mc127', 'peric': 'mc128',
              'pesso': 'mc129', 'pet': 'mc130', 'pontu': 'mc131', 'postul': 'mc132', 'prat': 'mc133', 'prec__': 'mc134',
              'pred': 'mc135', 'prejudic': 'mc136', 'prev': 'mc137', 'previd': 'mc138', 'produc': 'mc139',
              'proporc': 'mc140', 'propr': 'mc141', 'propriedad': 'mc142', 'provisor': 'mc143', 'public__': 'mc144',
              'pun': 'mc145', 'regim': 'mc146', 'reg__': 'mc147', 'registr': 'mc148', 'regul__': 'mc149',
              'regul': 'mc150', 'rei': 'mc151', 'reje': 'mc152', 'relat__': 'mc153', 'respons': 'mc154',
              'restr': 'mc155', 'ret': 'mc156', 'retific': 'mc157', 'salar': 'mc158', 'segur': 'mc159',
              'sindic': 'mc160', 'soc__': 'mc161', 'societ': 'mc162', 'substitu': 'mc163', 'sucess': 'mc164',
              'superfic': 'mc165', 'tabel': 'mc166', 'tax': 'mc167', 'temp': 'mc168', 'terren': 'mc169',
              'territor': 'mc170', 'titul': 'mc171', 'trasl': 'mc172', 'tribut': 'mc173', 'urban': 'mc174',
              'util': 'mc175', 'vacanc': 'mc176', 'val__': 'mc177', 'voluntar': 'mc178', 'xerocop': 'mc179',
              'abus': 'mc1', 'acessibil': 'mc2', 'aco': 'mc3', 'advocatici': 'mc4', 'agr': 'mc5', 'alienaca': 'mc6',
              'ampli': 'mc7', 'anterior': 'mc8', 'anual': 'mc9', 'anulabil': 'mc10', 'apreend': 'mc11',
              'aquisi': 'mc12', 'arpen/sp': 'mc13', 'ausent': 'mc14', 'autentic': 'mc15', 'averb': 'mc16',
              'bancar': 'mc17', 'boa_fe': 'mc18', 'bel': 'mc19', 'caus': 'mc20', 'cgjsp': 'mc21', 'civil': 'mc22',
              'cnd': 'mc23', 'comerci': 'mc24', 'compensatori': 'mc25', 'competenc': 'mc26', 'condic': 'mc27',
              'condomini': 'mc28', 'conjunt': 'mc29', 'consent': 'mc30', 'constituc': 'mc31', 'construc': 'mc32',
              'consult': 'mc33', 'contribuint': 'mc34', 'correc': 'mc35', 'correged': 'mc36', 'danific': 'mc37',
              'decis': 'mc38', 'declarator': 'mc39', 'demarcator': 'mc40', 'descri': 'mc41', 'destaqu': 'mc42',
              'dire': 'mc43', 'discriminator': 'mc44', 'doad': 'mc45', 'domini': 'mc46', 'duplic': 'mc47',
              'duvid': 'mc48', 'eleitor': 'mc49', 'emit': 'mc50', 'empresar': 'mc51', 'endossatar': 'mc52',
              'escritur': 'mc53', 'espec': 'mc54', 'especific': 'mc55', 'estatutar': 'mc56', 'examin': 'mc57',
              'excess': 'mc58', 'excepcional': 'mc59', 'execut': 'mc60', 'exigibil': 'mc61', 'expropriator': 'mc62',
              'extingu': 'mc63', 'extraordinar': 'mc64', 'falsific': 'mc65', 'famili': 'mc66', 'feit': 'mc67',
              'fiduciar': 'mc68', 'fili': 'mc69', 'final': 'mc70', 'formal': 'mc71', 'fracion': 'mc72',
              'fraudulent': 'mc73', 'garant': 'mc74', 'gerenc': 'mc75', 'gratuit': 'mc76', 'habitac': 'mc77',
              'herd': 'mc78', 'hipotecar': 'mc79', 'identific': 'mc80', 'imobiliar': 'mc81', 'imperfeit': 'mc82',
              'impos': 'mc83', 'inadimple': 'mc84', 'inscr': 'mc85', 'indefir': 'mc86', 'indigen': 'mc87',
              'indisponibil': 'mc88', 'individu': 'mc89', 'indivisibil': 'mc90', 'inexistenc': 'mc91',
              'infra_estrut': 'mc92', 'inici': 'mc93', 'inscrit': 'mc94', 'instituc': 'mc95', 'intempest': 'mc96',
              'interpost': 'mc97', 'intervenc': 'mc98', 'inviabil': 'mc99', 'irregular': 'mc100', 'judici': 'mc101',
              'jurisdic': 'mc102', 'justic': 'mc103', 'lavrat': 'mc104', 'legitim': 'mc105', 'locatic': 'mc106',
              'lote': 'mc107', 'ma_fe': 'mc108', 'magistrat': 'mc109', 'mand': 'mc110', 'mar': 'mc111',
              'matrimon': 'mc112', 'modal': 'mc113', 'morad': 'mc114', 'municipal': 'mc115', 'nacional': 'mc116',
              'necessar': 'mc117', 'notar': 'mc118', 'nulidad': 'mc119', 'obrigator': 'mc120', 'ofici': 'mc121',
              'ordinar': 'mc122', 'origin': 'mc123', 'padron': 'mc124', 'patrimoni': 'mc125', 'penal': 'mc126',
              'penhor': 'mc127', 'perit': 'mc128', 'pessoal': 'mc129', 'peticion': 'mc130', 'pont__': 'mc131',
              'postulator': 'mc132', 'pratic': 'mc133', 'precar': 'mc134', 'predi': 'mc135', 'prejudicial': 'mc136',
              'previ': 'mc137', 'previdenci': 'mc138', 'produt': 'mc139', 'proporcional': 'mc140', 'propri': 'mc141',
              'propriet': 'mc142', 'provisori': 'mc143', 'publ': 'mc144', 'punibil': 'mc145', 'regiment': 'mc146',
              'region': 'mc147', 'registrar': 'mc148', 'regulament': 'mc149', 'regular': 'mc150', 'reil': 'mc151',
              'rejeit': 'mc152', 'relatori': 'mc153', 'responsabil': 'mc154', 'restri': 'mc155', 'retenc': 'mc156',
              'retificaca': 'mc157', 'salari': 'mc158', 'seguranc': 'mc159', 'sindicat': 'mc160', 'soci': 'mc161',
              'societar': 'mc162', 'substitut': 'mc163', 'sucessori': 'mc164', 'superfici': 'mc165', 'tabelia': 'mc166',
              'taxa': 'mc167', 'tempor': 'mc168', 'terr': 'mc169', 'territorial': 'mc170', 'titular': 'mc171',
              'transl': 'mc172', 'tributar': 'mc173', 'urbanis': 'mc174', 'utiliz': 'mc175', 'vag': 'mc176',
              'validad': 'mc177', 'voluntari': 'mc178', 'xerox': 'mc179', 'actu': 'mc3', 'advog': 'mc4',
              'agricol': 'mc5', 'anulator': 'mc10', 'cnd/ins': 'mc23', 'constitucional': 'mc31', 'declaratori': 'mc39',
              'duplicat': 'mc47', 'especial': 'mc54', 'extint': 'mc63', 'filiator': 'mc69', 'institut': 'mc95',
              'judiciar': 'mc101', 'jurisdicion': 'mc102', 'municipi': 'mc115', 'obrigatori': 'mc120',
              'originar': 'mc123', 'pontual': 'mc131', 'precariedad': 'mc134', 'prejuiz': 'mc136',
              'previdenciar': 'mc138', 'tabelionat': 'mc166', 'taxat': 'mc167', 'actum': 'mc3', 'cnd/rf': 'mc23',
              'obrigatoriedad': 'mc120'}
MERGE_COUNT_DICT = copy.deepcopy(MERGE_DICT)
for k in MERGE_COUNT_DICT.keys():
    MERGE_COUNT_DICT[k] = 0

CLASSES2REMOVE = ['ces', 'ndeg', 'no', 'p/', 'sant', 'a']


STEMMING_MAP = {}

FREQUENCY_FILTERING_THRESHOLD = 5
COOCCURRENCE_THRESHOLD = 0.5
GROUPING_THRESHOLD = 1000  # 70
CLUSTERING_METHOD = 'KMeans'
KEEP_OTR_AS_CLUSTER = True


# Auxiliar functions:

def special_stem(word):
    """stem a word in a way more compatible with this application.

    Args:
        word (str): word to be stemmed

    Returns:
        str: stemmed word
    """
    stemmer = nltk.stem.RSLPStemmer()
    value = ''
    if word in SPECIAL_STEM_DICT:
        value = SPECIAL_STEM_DICT[word]
        SPECIAL_STEM_COUNT_DICT[word] = SPECIAL_STEM_COUNT_DICT[word] + 1
    else:
        value = stemmer.stem(word)

    if unidecode.unidecode(value).encode('ascii').decode('utf-8') in MERGE_DICT:
        MERGE_COUNT_DICT[unidecode.unidecode(value).encode('ascii').decode('utf-8')] = MERGE_COUNT_DICT[
            unidecode.unidecode(
                value).encode(
                'ascii').decode(
                'utf-8')] + 1
        value = MERGE_DICT[unidecode.unidecode(
            value).encode('ascii').decode('utf-8')]
    return value


# Here begins the actual processing:
print('VERBERT PROJECT')
print('|=> Ontological Adjustment --v=0.1')

# Load corpus:
print('    |=> Loading Corpus:')
with open(FIELDS_FILE_NAME, 'rb') as f:
    fileds = pickle.load(f)
Processo = namedlist('Processo', fileds)
with open(CORPUS_FILE_NAME, 'rb') as f:
    corpus = pickle.load(f)
password = getpass("        |=> Password:")
sys.stdout.write('\033[F')
print("        |=> Decrypting Corpus...")
crypto = Naive_Criptography()
corpus = crypto.naive_corpus_decryption(password, corpus)
print('        |=> Corpus Loaded')

# Split classes into their components, using the special stemming function
print('    |=> Spliting Classes:')
temp_corpus = []
cont = 0
original_classes_list = []
original_classes_frequencies = {}
for processo in tqdm(corpus, desc='         '):
    new_class_set = []

    for verbete in processo.verbetacao:
        special_split = False

        for mwv in MULTI_WORD_CLASS_LIST:
            if mwv in verbete:
                special_split = mwv

        if verbete not in MULTI_WORD_CLASS_LIST\
                and ('-' in verbete or '???' in verbete or '_' in verbete)\
                and not special_split:
            splitted_class = re.split(
                '(?<!n??)(?<!cg)(?<!\d)[-???_]+|[-???_]+(?=[0-9]+[??????]{1})|(?<=\d)[-???_]+(?=\D)', verbete)
            temp_list = []

            for nv in splitted_class:
                if nv in TO_REMOVE_FIRST_CHAR:
                    nv = nv[1:]
                if nv in TO_REMOVE_LAST_CHAR:
                    nv = nv[:len(nv) - 1]
                if not ((nv in STOPWORDS) or (re.search('[0-9]', nv)) or (re.search(ROMAN_NUMERAL_REGEX, nv))):
                    nv_orig = nv
                    nv = special_stem(nv)
                    nv = unidecode.unidecode(nv).encode('ascii')
                    nv = nv.decode('utf-8')
                    temp_list.append(nv)
                    if nv in STEMMING_MAP:
                        if nv_orig not in STEMMING_MAP[nv]:
                            STEMMING_MAP[nv].append(nv_orig)
                    else:
                        STEMMING_MAP[nv] = [nv_orig]

            splitted_class = temp_list
            new_class_set = new_class_set + splitted_class

        elif special_split and ('-' in verbete or '???' in verbete or '_' in verbete):
            special_splitted_array = re.split(special_split, verbete)
            if special_splitted_array[0] == '' and special_splitted_array[1] == '':
                if not (verbete in STOPWORDS or re.search('[0-9]', verbete) or re.search(ROMAN_NUMERAL_REGEX, verbete)):
                    if verbete in TO_REMOVE_FIRST_CHAR:
                        verbete = verbete[1:]
                    if verbete in TO_REMOVE_LAST_CHAR:
                        verbete = verbete[:len(verbete) - 1]
                    verbete_orig = verbete
                    verbete = special_stem(verbete)
                    verbete = unidecode.unidecode(verbete).encode('ascii')
                    verbete = verbete.decode('utf-8')
                    new_class_set.append(verbete)
                    if verbete in STEMMING_MAP:
                        if verbete_orig not in STEMMING_MAP[verbete]:
                            STEMMING_MAP[verbete].append(verbete_orig)
                    else:
                        STEMMING_MAP[verbete] = [verbete_orig]
            else:
                splitted_class = re.split('(?<!n??)(?<!cg)(?<!\d)[-???_]+|[-???_]+(?=[0-9]+[??????]{1})|(?<=\d)[-???_]+(?=\D)',
                                          special_splitted_array[0])
                temp_list = []
                for nv in splitted_class:
                    ROMAN_NUMERAL_REGEX = '^(?=[mdclxvi])m*(c[md]|d?c*)(x[cl]|l?x*)(i[xv]|v?i*)$'
                    if nv in TO_REMOVE_FIRST_CHAR:
                        nv = nv[1:]
                    if nv in TO_REMOVE_LAST_CHAR:
                        nv = nv[:len(nv) - 1]
                    if not ((nv in STOPWORDS) or (re.search('[0-9]', nv)) or (re.search(ROMAN_NUMERAL_REGEX, nv))):
                        nv_orig = nv
                        nv = special_stem(nv)
                        nv = unidecode.unidecode(nv).encode('ascii')
                        nv = nv.decode('utf-8')
                        temp_list.append(nv)
                        new_class_set.append(nv)
                        if nv in STEMMING_MAP:
                            if nv_orig not in STEMMING_MAP[nv]:
                                STEMMING_MAP[nv].append(nv_orig)
                        else:
                            STEMMING_MAP[nv] = [nv_orig]

                if not (special_split in STOPWORDS or re.search('[0-9]', special_split) or re.search(
                        ROMAN_NUMERAL_REGEX, special_split)):
                    if special_split in TO_REMOVE_FIRST_CHAR:
                        special_split = special_split[1:]
                    if special_split in TO_REMOVE_LAST_CHAR:
                        special_split = special_split[:len(special_split) - 1]
                    special_split_orig = special_split
                    special_split = special_stem(special_split)
                    special_split = unidecode.unidecode(
                        special_split).encode('ascii')
                    special_split = special_split.decode('utf-8')
                    new_class_set.append(special_split)
                    if special_split in STEMMING_MAP:
                        if special_split_orig not in STEMMING_MAP[special_split]:
                            STEMMING_MAP[special_split].append(
                                special_split_orig)
                    else:
                        STEMMING_MAP[special_split] = [special_split_orig]

                splitted_class = re.split('(?<!n??)(?<!cg)(?<!\d)[-???_]+|[-???_]+(?=[0-9]+[??????]{1})|(?<=\d)[-???_]+(?=\D)',
                                          special_splitted_array[1])
                temp_list = []
                for nv in splitted_class:
                    ROMAN_NUMERAL_REGEX = '^(?=[mdclxvi])m*(c[md]|d?c*)(x[cl]|l?x*)(i[xv]|v?i*)$'
                    if nv in TO_REMOVE_FIRST_CHAR:
                        nv = nv[1:]
                    if nv in TO_REMOVE_LAST_CHAR:
                        nv = nv[:len(nv) - 1]
                    if not ((nv in STOPWORDS) or (re.search('[0-9]', nv)) or (re.search(ROMAN_NUMERAL_REGEX, nv))):
                        nv_orig = nv
                        nv = special_stem(nv)
                        nv = unidecode.unidecode(nv).encode('ascii')
                        nv = nv.decode('utf-8')
                        temp_list.append(nv)
                        new_class_set.append(nv)

                        if nv in STEMMING_MAP:
                            if nv_orig not in STEMMING_MAP[nv]:
                                STEMMING_MAP[nv].append(nv_orig)
                        else:
                            STEMMING_MAP[nv] = [nv_orig]

        else:
            if not (verbete in STOPWORDS or re.search('[0-9]', verbete) or re.search(ROMAN_NUMERAL_REGEX, verbete)):
                if verbete in TO_REMOVE_FIRST_CHAR:
                    verbete = verbete[1:]
                if verbete in TO_REMOVE_LAST_CHAR:
                    verbete = verbete[:len(verbete) - 1]
                verbete_orig = verbete
                verbete = special_stem(verbete)
                verbete = unidecode.unidecode(verbete).encode('ascii')
                verbete = verbete.decode('utf-8')
                new_class_set.append(verbete)
                if verbete in STEMMING_MAP:
                    if verbete_orig not in STEMMING_MAP[verbete]:
                        STEMMING_MAP[verbete].append(verbete_orig)
                else:
                    STEMMING_MAP[verbete] = [verbete_orig]

    if len(new_class_set) > 0:
        processo.verbetacao = new_class_set
        temp_corpus.append(processo)
        for c in new_class_set:
            if c in original_classes_frequencies:
                original_classes_frequencies[c] = original_classes_frequencies[c] + 1
            else:
                original_classes_frequencies[c] = 1
    cont = cont + 1

corpus = temp_corpus


# Filter the classes whose frequency is less than the threshold from the corpus and adjacent structures
print('    |=> Frequency Filtering:')
temp_ocf = copy.deepcopy(original_classes_frequencies)
for verb in original_classes_frequencies:
    if original_classes_frequencies[verb] <= FREQUENCY_FILTERING_THRESHOLD:
        del temp_ocf[verb]

original_classes_frequencies = temp_ocf

for processo in tqdm(corpus, desc='         '):
    unique_verbetacao = [processo.verbetacao[i] for i in range(len(processo.verbetacao)) if
                         processo.verbetacao[i] not in processo.verbetacao[:max(i, 0)]]
    new_verbetacao = copy.deepcopy(unique_verbetacao)
    for verbete in unique_verbetacao:
        if verbete not in original_classes_frequencies.keys():
            new_verbetacao.remove(verbete)
            if verbete in STEMMING_MAP.keys():
                del STEMMING_MAP[verbete]
    processo.verbetacao = new_verbetacao
temp_corpus = copy.deepcopy(corpus)
for processo in corpus:
    if len(processo.verbetacao) == 0:
        temp_corpus.remove(processo)
corpus = copy.deepcopy(temp_corpus)

# Remove the spurious classes manually defined
for c2r in CLASSES2REMOVE:
    if c2r in STEMMING_MAP.keys():
        del STEMMING_MAP[c2r]
        del original_classes_frequencies[c2r]
    else:
        print('Class not removed: ' + c2r)
    for processo in corpus:
        c2rin = False
        new_verb = processo.verbetacao
        if c2r in new_verb:
            c2rin = True
            new_verb.remove(c2r)
        processo.verbetacao = new_verb

# Check that the classes have been removed correctly
for c2r in CLASSES2REMOVE:
    for processo in corpus:
        if c2r in processo.verbetacao:
            print('Alert! Class not deleted: ' + c2r)


# Search for corrupted processes and spare classes
cont = 0
for processo in corpus:
    if len(processo.verbetacao) == 0:
        print('Alert! Null entry!')
    for verb in processo.verbetacao:
        cont = cont + 1
        if verb not in STEMMING_MAP.keys() or verb not in original_classes_frequencies.keys():
            print('Alert:' + str(verb))

# Save the frequency filtered stemming map to disk
setemming_map_file = open('stemming_map.txt', 'w')
for verb in sorted(list(STEMMING_MAP.keys())):
    setemming_map_file.write(str(verb) + ': ' + str(STEMMING_MAP[verb]) + '\n')

setemming_map_file.close()

# Filter classes by co-occurrence, setting up a hierarchical structure according to these relationships
print('    |=> Co-occurrence Filtering:')
# Fill in the Co-occurrence matrix
class2index = copy.deepcopy(STEMMING_MAP)
index2class = {}
cont = 0
for cl in class2index:
    class2index[cl] = cont
    index2class[cont] = cl
    cont = cont + 1
coocc_matrix = np.zeros((len(STEMMING_MAP), len(STEMMING_MAP)))
for processo in tqdm(corpus, desc='         '):
    for verb1 in processo.verbetacao:
        for verb2 in processo.verbetacao:
            if verb1 != verb2:
                coocc_matrix[class2index[verb1]][class2index[verb2]] = coocc_matrix[class2index[verb1]][
                    class2index[verb2]] + (1 / (
                        2 * original_classes_frequencies[verb2]))
                coocc_matrix[class2index[verb2]][class2index[verb1]] = coocc_matrix[class2index[verb2]][
                    class2index[verb1]] + (1 / (
                        2 * original_classes_frequencies[verb1]))

# Assemble the hierarchical structure of the classes from the co-occurrence matrix
sub_stemming_map = {}
verb_tree = []
node0 = Node('Kollemata')
for i in range(len(STEMMING_MAP)):
    verb_tree.append(Node(index2class[i]))

for l in tqdm(range(len(coocc_matrix)), desc='         '):
    filho = index2class[l]
    coocc = np.amax(coocc_matrix[:, l])
    pai = index2class[np.where(coocc_matrix[:, l] == coocc)[0][0]]
    if coocc > COOCCURRENCE_THRESHOLD and verb_tree[np.where(coocc_matrix[:, l] == coocc)[0][0]] not in verb_tree[l].children:
        verb_tree[l].parent = verb_tree[np.where(
            coocc_matrix[:, l] == coocc)[0][0]]
        sub_stemming_map[filho] = STEMMING_MAP[filho]
        del STEMMING_MAP[filho]
    else:
        verb_tree[l].parent = node0

# Update the corpus
for processo in corpus:
    new_verbetacao = []
    for verb in processo.verbetacao:
        new_verbetacao.append(verb_tree[class2index[verb]])

    processo.verbetacao = new_verbetacao

# Computes the appearance frequencies of the classes from the first level of the hierarchical structure
fst_lvl_classes_frequencies = {}
fst_lvl_classes_frequencies_no_repetition = {}
for processo in corpus:
    unique_verb = []
    for verb in processo.verbetacao:
        if verb.depth == 1:
            fstlvl_verb = verb.name
        else:
            fstlvl_verb = verb.ancestors[1].name
        unique_verb.append(fstlvl_verb)
        if fstlvl_verb in fst_lvl_classes_frequencies:
            fst_lvl_classes_frequencies[fstlvl_verb] = fst_lvl_classes_frequencies[fstlvl_verb] + 1
        else:
            fst_lvl_classes_frequencies[fstlvl_verb] = 1
    unique_verb = np.unique(unique_verb)
    for uverb in unique_verb:
        if uverb in fst_lvl_classes_frequencies_no_repetition:
            fst_lvl_classes_frequencies_no_repetition[uverb] = fst_lvl_classes_frequencies_no_repetition[uverb] + 1
        else:
            fst_lvl_classes_frequencies_no_repetition[uverb] = 1

# Group the top classes of the structure with low occurrence under the class 'others'
print('    |=> Low Frequency classes grouping:')
low_frequency_classes = {}
lfc_occ = 0
lfc_occ_no_rep = 0
stemming_map_copy = copy.deepcopy(STEMMING_MAP)
for clss in tqdm(stemming_map_copy, desc='         '):
    if fst_lvl_classes_frequencies[clss] < GROUPING_THRESHOLD:
        low_frequency_classes[clss] = STEMMING_MAP[clss]
        del STEMMING_MAP[clss]
        lfc_occ = lfc_occ + fst_lvl_classes_frequencies[clss]
        lfc_occ_no_rep = lfc_occ_no_rep + \
            fst_lvl_classes_frequencies_no_repetition[clss]
        del fst_lvl_classes_frequencies[clss]
        del fst_lvl_classes_frequencies_no_repetition[clss]

otr = Node('otr', parent=node0)
for processo in tqdm(corpus, desc='         '):
    new_verb = []
    otr_cont = 0
    for verb in processo.verbetacao:
        if verb.name in low_frequency_classes:
            otr_cont = otr_cont + 1
            verb.parent = otr
            new_verb.append(verb)
        else:
            new_verb.append(verb)
    processo.verbetacao = new_verb
    if otr_cont > 0:
        lfc_occ_no_rep = lfc_occ_no_rep - (otr_cont - 1)


# Cluster the classes at the top of the structure to generate the super classes
print('    |=> Clustering_classes:')
# Create a simplified version of the corpus for clusteing
clustering_corpus = []
for processo in corpus:
    verbet = processo.verbetacao
    clust_verbet = []
    for verb in verbet:
        if verb.depth == 1:
            fstlvl_verb = verb.name
        else:
            fstlvl_verb = verb.ancestors[1].name
        if fstlvl_verb != 'otr':
            clust_verbet.append(fstlvl_verb)
    clust_verbet = np.unique(clust_verbet)
    if len(clust_verbet) > 0:
        clustering_corpus.append(clust_verbet)

# Invert the corpus, creating an appropriate version for clustering the classes and not the documents:
class_clustering_corpus = []

ccc_indexing = {}
ccc_couter_indexing = {}
ccc_max_index = 0
for doc_number in range(len(clustering_corpus)):
    for clss in clustering_corpus[doc_number]:
        if clss not in ccc_indexing:
            ccc_indexing[clss] = ccc_max_index
            ccc_couter_indexing[ccc_max_index] = clss
            ccc_max_index = ccc_max_index + 1
            class_clustering_corpus.append([])
        class_clustering_corpus[ccc_indexing[clss]].append(str(doc_number))

for cont in range(len(class_clustering_corpus)):
    class_clustering_corpus[cont] = ' '.join(
        word for word in class_clustering_corpus[cont])

# Create the tf.idf features to representing the classes and compact the resulting representations
vectorizer = TfidfVectorizer()
classes_tfidf_matrix = vectorizer.fit_transform(class_clustering_corpus)
classes_clustering_index = {}
ctm_backup = copy.deepcopy(classes_tfidf_matrix)
for nc_red in [50]:
    classes_tfidf_matrix = copy.deepcopy(ctm_backup)
    if nc_red != 0:
        tsvd = TruncatedSVD(n_components=nc_red).fit(classes_tfidf_matrix)
        classes_tfidf_matrix = tsvd.transform(classes_tfidf_matrix)
    # Cluster the classes using k-means
    kmeans = KMeans(n_clusters=25, random_state=928300,
                    n_init=100, max_iter=1000).fit(classes_tfidf_matrix)
    # Compute silhouette scores
    sil_score = metrics.silhouette_score(
        classes_tfidf_matrix, kmeans.labels_, metric='euclidean')
    sil_score_2 = metrics.silhouette_score(
        ctm_backup, kmeans.labels_, metric='euclidean')

    clusterized_data = []
    for verbt in clustering_corpus:
        c_verbt = []
        for verb in verbt:
            c_verbt.append(kmeans.labels_[ccc_indexing[verb]])
        clusterized_data.append(c_verbt)
    clusters = []
    cluster_lens = []
    for i in range(np.amax(kmeans.labels_) + 1):
        clusters.append([])
        cluster_lens.append([0, 0])
    for j in range(len(kmeans.labels_)):
        clusters[kmeans.labels_[j]].append(ccc_couter_indexing[j])
        cluster_lens[kmeans.labels_[j]
                     ][0] = cluster_lens[kmeans.labels_[j]][0] + 1
    for c_verbt in clusterized_data:
        for verb in c_verbt:
            cluster_lens[verb][1] = cluster_lens[verb][1] + 1

    clust_occ_in_corpus = [x[1] for x in cluster_lens]
    mean_occ_clust_class = np.mean(clust_occ_in_corpus)
    xi2 = 0
    for coic in clust_occ_in_corpus:
        xi2 = xi2 + ((coic - mean_occ_clust_class) ** 2)
    # Save the grouping results to disk
    sys.stdout = open('clust' + '_' + CLUSTERING_METHOD +
                      '_' + str(nc_red) + '.txt', 'w')
    print(kmeans.labels_)
    cont = 0
    for a in clusters:
        cluster_sum = 0
        print(str(cont) + ' : ' + str(cluster_lens[cont]))
        for b in a:
            print(b)
            cluster_sum = cluster_sum + \
                fst_lvl_classes_frequencies_no_repetition[b]
        cont = cont + 1
        print(cluster_sum)

    print(
        str(sil_score).replace('.', ',') + '\t' + str(sil_score_2).replace('.', ',') + '\t' + str(np.sqrt(xi2)).replace(
            '.', ','))
    print('silhouette_score: ' + str(sil_score))
    print('silhouette_score2: ' + str(sil_score_2))
    print('xi_score: ' + str(np.sqrt(xi2)))
    sys.stdout = sys.__stdout__

    tsvd = TruncatedSVD(n_components=2).fit(classes_tfidf_matrix)
    tsvd_2d = tsvd.transform(classes_tfidf_matrix)

    y_kmeans = kmeans.predict(classes_tfidf_matrix)

    plt.scatter(tsvd_2d[:, 0], tsvd_2d[:, 1],
                c=y_kmeans, s=50, cmap='nipy_spectral')
    plt.title('Clustering com ' + CLUSTERING_METHOD +
              ' e redu????o pr??via para ' + str(nc_red) + ' dim')
    if nc_red == 0:
        plt.title('Clustering com ' + CLUSTERING_METHOD +
                  'sem redu????o dimensional pr??via')
    plt.savefig('clust_' + CLUSTERING_METHOD + '_' + str(nc_red) + '.jpg')
    plt.clf()


# Insert the resulting clusters at the top of the hierarchy
map_class_1to0 = {}
clust_id = 0
for clust in clusters:
    for fst_lvl_class in clust:
        if fst_lvl_class in map_class_1to0:
            print('Alert: First instance class assigned to more than one cluster.')
        else:
            map_class_1to0[fst_lvl_class] = clust_id
    clust_id = clust_id + 1

if (KEEP_OTR_AS_CLUSTER):
    map_class_1to0['otr'] = clust_id + 1


cluster_nodes = {}
for i in range(clust_id + 2):
    cluster_nodes[i] = Node('clust_' + str(i), parent=node0)

temp_corpus = copy.deepcopy(corpus)
proc_id = 0
for processo in tqdm(corpus, desc='         '):
    new_verbetacao = []
    for verb in processo.verbetacao:
        if (len(verb.ancestors) > 1 and 'clust' not in verb.ancestors[1].name) or (
                len(verb.ancestors) == 1 and 'clust' not in verb.name):
            new_verb = verb
            if len(verb.ancestors) == 1:
                verb_clust_id = map_class_1to0[new_verb.name]
                new_verb.parent = cluster_nodes[verb_clust_id]
            else:
                verb_clust_id = map_class_1to0[new_verb.ancestors[1].name]
                new_verb.ancestors[1].parent = cluster_nodes[verb_clust_id]
            new_verbetacao.append(new_verb)
        else:
            new_verb = verb
            new_verbetacao.append(new_verb)
    temp_corpus[proc_id].verbetacao = new_verbetacao
    proc_id = proc_id + 1

# Saves the complete corpus to disk and the resulting ontology/hierarchy to disk
corpus = copy.deepcopy(temp_corpus)
with open('koll_corpus_KM50_complete.pkl', 'wb') as f:
    pickle.dump(corpus, f)
with open('koll_fields_KM50_complete.pkl', 'wb') as f:
    pickle.dump(fileds, f)
orig_stdout = sys.stdout
sys.stdout = open('ontology_KM50.txt', 'w')
print(RenderTree(node0))
sys.stdout = orig_stdout

# Create a version of the corpus using only the top classes of the hierarchical structure
corpus_lvl0only = copy.deepcopy(corpus)
for processo in corpus_lvl0only:
    lvl0_verbs = []
    for verb in processo.verbetacao:
        if not 'clust' in verb.ancestors[1].name:
            print('Alerta: classe de nivel 0 n??o correspondente a um cluster.')
        lvl0_verbs.append(verb.ancestors[1].name)
    lvl0_verbs = np.unique(lvl0_verbs).tolist()
    processo.verbetacao = lvl0_verbs

# Save this version to disk
with open('koll_corpus_KM50_lvl0.pkl', 'wb') as f:
    pickle.dump(corpus_lvl0only, f)
with open('koll_fields_KM50_lvl0.pkl', 'wb') as f:
    pickle.dump(fileds, f)
