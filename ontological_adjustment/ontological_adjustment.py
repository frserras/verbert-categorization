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

LEGAL_MULTIWORD_EXPRESSIONS = ['a_quo', 'abolitio_criminis', 'ad_quem', 'advocacia-geral_da_união', 'amicus_curiae',
                               'bis_in_idem', 'caso_fortuito', 'causa_de_pedir', 'cláusula_pétrea', 'coisa_julgada',
                               'ex_officio', 'ex_nunc', 'ex_tunc', 'extradição', 'força_maior', 'habeas_corpus',
                               'habeas_data', 'iniciativa_de_lei', 'kompetenz-kompetenz', 'negócio_jurídico',
                               'nexo_causal', 'ordem_dos_advogados_do_brasil', 'partido_político', 'queixa_crime',
                               'quinto_constitucional', 'transitar_em_julgado', 'distrito_policial',
                               'acessorium_sequitur_principale', 'ad_corpus', 'ad_impossibilia_nemo_tenetur',
                               'ad_mensuram', 'advocacia_geral_da_união', 'ad_corpus', 'causa_petendi', 'amicus_curiae',
                               'ano_novo', 'a-quo', 'abolitio-criminis', 'ad-quem', 'advocacia-geral-da-união',
                               'amicus-curiae', 'bis-in-idem', 'caso-fortuito', 'causa-de-pedir', 'cláusula-pétrea',
                               'coisa-julgada', 'de-ofício', 'ex-officio', 'ex-nunc', 'ex-tunc', 'extradição',
                               'força-maior', 'habeas-corpus', 'habeas-data', 'iniciativa-de-lei',
                               'kompetenz-kompetenz', 'negócio-jurídico', 'nexo-causal',
                               'ordem-dos-advogados-do-brasil', 'partido-político', 'queixa-crime',
                               'quinto-constitucional', 'transitar-em-julgado', 'distrito-policial',
                               'acessorium-sequitur-principale', 'ad-corpus', 'ad-impossibilia-nemo-tenetur',
                               'ad-mensuram', 'advocacia-geral-da-união', 'ad-corpus', 'causa-petendi', 'amicus-curiae',
                               'ano-novo']
GENERAL_MULTIWORD_EXPRESSIONS = ['abaixo_assinado', 'abaixo_assinados', 'aca_de_açucares', 'alto_falante',
                                 'alto_falantes', 'anos_luz', 'anti_retrovirais', 'anti_retroviral', 'ar_condicionado',
                                 'ar_condicionados', 'arco_íris', 'ares_condicionados', 'atividade_fim',
                                 'atividades_fim', 'auto_imune', 'auto_imunes', 'banho_maria', 'batata_doce',
                                 'batatas_doce', 'bate_boca', 'bate_papo', 'bate_papos', 'beija_flor', 'beija_flores',
                                 'bem_estar', 'bem_sucedida', 'bem_sucedidas', 'bem_sucedido', 'bem_sucedidos',
                                 'bem_vinda', 'bem_vindas', 'bem_vindo', 'bem_vindos', 'boa_fé', 'boas_vindas',
                                 'boca_de_urna', 'caixa_preta', 'caixa_pretas', 'caixas_preta', 'caixas_pretas',
                                 'cana_de_açúcar', 'canas_de_açúcar', 'capim_elefante', 'carro_chefe', 'carro_forte',
                                 'carros_chefe', 'carros_forte', 'case_control', 'cd_rom', 'cd_roms', 'censo_designada',
                                 'censo_designadas', 'censo_designado', 'censo_designados', 'coca_cola', 'coca_colas',
                                 'contra_ataque', 'contra_ataques', 'cor_de_rosa', 'corpo_de_prova', 'corpo_de_provas',
                                 'corpos_de_provas', 'cross_sectional', 'curta_metragem', 'curta_metragens',
                                 'curto_circuito', 'curto_circuitos', 'célula_tronco', 'células_tronco', 'data_base',
                                 'data_bases', 'decreto_lei', 'decreto_leis', 'decretos_lei', 'decretos_leis',
                                 'desvio_padrão', 'desvio_padrões', 'desvios_padrão', 'dia_a_dia', 'dona_de_casa',
                                 'donas_de_casa', 'double_blind', 'duplo_cego', 'e_mail', 'e_mails', 'ene_ésima',
                                 'ene_ésimas', 'erva_mate', 'ervas_mate', 'follow_up', 'free_lance', 'free_lancers',
                                 'free_lances', 'grã_bretanha', 'guarda_costas', 'guarda_roupa', 'guarda_roupas',
                                 'high_risk', 'hip_hop', 'i_ésima', 'i_ésimas', 'infra_estrutura', 'infra_estruturas',
                                 'inter_relacionada', 'inter_relacionadas', 'inter_relacionado', 'inter_relacionados',
                                 'inter_relacionamentos', 'inter_relações', 'key_word', 'key_words', 'know_how',
                                 'long_term', 'longa_metragem', 'longa_metragens', 'lua_de_mel', 'luas_de_mel',
                                 'lugar_comum', 'língua_alvo', 'línguas_alvo', 'mais_valia', 'mal_entendido',
                                 'mal_estar', 'mata_mata', 'matéria_prima', 'matéria_primas', 'mau_estar',
                                 'meia_atacante', 'meia_atacantes', 'meia_idade', 'meia_noite', 'meia_vida',
                                 'meia_vidas', 'meio_ambiente', 'meio_campo', 'meio_campos', 'meio_dia', 'meio_termo',
                                 'meio_termos', 'meios_campo', 'meios_termos', 'mercedes_benz', 'meta_analysis',
                                 'meu_entendido', 'má_fé', 'mão_de_obra', 'n_ésima', 'n_ésimas', 'obra_prima',
                                 'obras_primas', 'oitava_de_final', 'oitavas_de_final', 'on_line', 'p_valor',
                                 'padrão_ouro', 'padrões_ouro', 'palavra_chave', 'palavras_chave', 'palavras_chaves',
                                 'pan_americana', 'pan_americanas', 'pan_americanos', 'pau_brasil', 'paus_brasil',
                                 'paus_brasis', 'pay_per_view', 'pay_per_views', 'peso_pesado', 'pesos_pesados',
                                 'pesquisa_ação', 'pesquisas_ação', 'placebo_controlled', 'population_based',
                                 'população_alvo', 'populações_alvo', 'porta_aviões', 'porta_enxerto', 'porta_enxertos',
                                 'porta_malas', 'porta_voz', 'porta_vozes', 'post_mortem', 'primeira_dama',
                                 'primeira_damas', 'primeiras_damas', 'pronto_atendimento', 'pronto_atendimentos',
                                 'pronto_socorro', 'pronto_socorros', 'prontos_atendimentos', 'prontos_socorros',
                                 'pré_frontal', 'pré_natais', 'pré_natal', 'pré_requisito', 'pré_requisitos',
                                 'pára_quedas', 'pôr_do_sol', 'quadro_negro', 'quadros_negros', 'quarta_de_final',
                                 'quarta_feira', 'quartas_de_final', 'quartas_feiras', 'quarteis_generais',
                                 'quartel_general', 'quebra_cabeça', 'quebra_cabeças', 'qui_quadrado', 'quinta_feira',
                                 'quintas_feiras', 'raio_x', 'raios_x', 'rá_tim_bum', 'salva_vidas',
                                 'secretário_executivo', 'secretários_executivos', 'segunda_feira', 'segundas_feiras',
                                 'sexta_feira', 'sexta_feiras', 'sextas_feiras', 'short_term', 'springer_verlag',
                                 'super_homem', 'terça_feira', 'terças_feiras', 'todo_poderoso', 'ultra_som',
                                 'ultra_sonografia', 'ultra_sonografias', 'ultra_sonográfico', 'ultra_sonográficos',
                                 'ultra_sons', 'ultra_sônica', 'ultra_sônico', 'ultra_sônicos', 'ultra⁻sônicas',
                                 'vice_versa', 'villa_lobos', 'vis_à_vis', 'x_man', 'x_men', 'x_ray', 'x_rays',
                                 'abaixo-assinado', 'abaixo-assinados', 'aca-de-açucares', 'alto-falante',
                                 'alto-falantes', 'anos-luz', 'anti-retrovirais', 'anti-retroviral', 'ar-condicionado',
                                 'ar-condicionados', 'arco-íris', 'ares-condicionados', 'atividade-fim',
                                 'atividades-fim', 'auto-imune', 'auto-imunes', 'banho-maria', 'batata-doce',
                                 'batatas-doce', 'bate-boca', 'bate-papo', 'bate-papos', 'beija-flor', 'beija-flores',
                                 'bem-estar', 'bem-sucedida', 'bem-sucedidas', 'bem-sucedido', 'bem-sucedidos',
                                 'bem-vinda', 'bem-vindas', 'bem-vindo', 'bem-vindos', 'boa-fé', 'boas-vindas',
                                 'boca-de-urna', 'caixa-preta', 'caixa-pretas', 'caixas-preta', 'caixas-pretas',
                                 'cana-de-açúcar', 'canas-de-açúcar', 'capim-elefante', 'carro-chefe', 'carro-forte',
                                 'carros-chefe', 'carros-forte', 'case-control', 'cd-rom', 'cd-roms', 'censo-designada',
                                 'censo-designadas', 'censo-designado', 'censo-designados', 'coca-cola', 'coca-colas',
                                 'contra-ataque', 'contra-ataques', 'cor-de-rosa', 'corpo-de-prova', 'corpo-de-provas',
                                 'corpos-de-provas', 'cross-sectional', 'curta-metragem', 'curta-metragens',
                                 'curto-circuito', 'curto-circuitos', 'célula-tronco', 'células-tronco', 'data-base',
                                 'data-bases', 'decreto-lei', 'decreto-leis', 'decretos-lei', 'decretos-leis',
                                 'desvio-padrão', 'desvio-padrões', 'desvios-padrão', 'dia-a-dia', 'dona-de-casa',
                                 'donas-de-casa', 'double-blind', 'duplo-cego', 'e-mail', 'e-mails', 'ene-ésima',
                                 'ene-ésimas', 'erva-mate', 'ervas-mate', 'follow-up', 'free-lance', 'free-lancers',
                                 'free-lances', 'grã-bretanha', 'guarda-costas', 'guarda-roupa', 'guarda-roupas',
                                 'high-risk', 'hip-hop', 'i-ésima', 'i-ésimas', 'infra-estrutura', 'infra-estruturas',
                                 'inter-relacionada', 'inter-relacionadas', 'inter-relacionado', 'inter-relacionados',
                                 'inter-relacionamentos', 'inter-relações', 'key-word', 'key-words', 'know-how',
                                 'long-term', 'longa-metragem', 'longa-metragens', 'lua-de-mel', 'luas-de-mel',
                                 'lugar-comum', 'língua-alvo', 'línguas-alvo', 'mais-valia', 'mal-entendido',
                                 'mal-estar', 'mata-mata', 'matéria-prima', 'matéria-primas', 'mau-estar',
                                 'meia-atacante', 'meia-atacantes', 'meia-idade', 'meia-noite', 'meia-vida',
                                 'meia-vidas', 'meio-ambiente', 'meio-campo', 'meio-campos', 'meio-dia', 'meio-termo',
                                 'meio-termos', 'meios-campo', 'meios-termos', 'mercedes-benz', 'meta-analysis',
                                 'meu-entendido', 'má-fé', 'mão-de-obra', 'n-ésima', 'n-ésimas', 'obra-prima',
                                 'obras-primas', 'oitava-de-final', 'oitavas-de-final', 'on-line', 'p-valor',
                                 'padrão-ouro', 'padrões-ouro', 'palavra-chave', 'palavras-chave', 'palavras-chaves',
                                 'pan-americana', 'pan-americanas', 'pan-americanos', 'pau-brasil', 'paus-brasil',
                                 'paus-brasis', 'pay-per-view', 'pay-per-views', 'peso-pesado', 'pesos-pesados',
                                 'pesquisa-ação', 'pesquisas-ação', 'placebo-controlled', 'population-based',
                                 'população-alvo', 'populações-alvo', 'porta-aviões', 'porta-enxerto', 'porta-enxertos',
                                 'porta-malas', 'porta-voz', 'porta-vozes', 'post-mortem', 'primeira-dama',
                                 'primeira-damas', 'primeiras-damas', 'pronto-atendimento', 'pronto-atendimentos',
                                 'pronto-socorro', 'pronto-socorros', 'prontos-atendimentos', 'prontos-socorros',
                                 'pré-frontal', 'pré-natais', 'pré-natal', 'pré-requisito', 'pré-requisitos',
                                 'pára-quedas', 'pôr-do-sol', 'quadro-negro', 'quadros-negros', 'quarta-de-final',
                                 'quarta-feira', 'quartas-de-final', 'quartas-feiras', 'quarteis-generais',
                                 'quartel-general', 'quebra-cabeça', 'quebra-cabeças', 'qui-quadrado', 'quinta-feira',
                                 'quintas-feiras', 'raio-x', 'raios-x', 'rá-tim-bum', 'salva-vidas',
                                 'secretário-executivo', 'secretários-executivos', 'segunda-feira', 'segundas-feiras',
                                 'sexta-feira', 'sexta-feiras', 'sextas-feiras', 'short-term', 'springer-verlag',
                                 'super-homem', 'terça-feira', 'terças-feiras', 'todo-poderoso', 'ultra-som',
                                 'ultra-sonografia', 'ultra-sonografias', 'ultra-sonográfico', 'ultra-sonográficos',
                                 'ultra-sons', 'ultra-sônica', 'ultra-sônico', 'ultra-sônicos', 'ultra⁻sônicas',
                                 'vice-versa', 'villa-lobos', 'vis-à-vis', 'x-man', 'x-men', 'x-ray', 'x-rays']
MULTI_WORD_CLASS_LIST = LEGAL_MULTIWORD_EXPRESSIONS + GENERAL_MULTIWORD_EXPRESSIONS
ROMAN_NUMERAL_REGEX = '^(?=[mdclxvi])m*(c[md]|d?c*)(x[cl]|l?x*)(i[xv]|v?i*)$'

PLACE_NAMES = ["são-paulo", "guarulhos", "campinas", "são-bernardo-do-campo",
               "são-josé-dos-campos", "santo-andré", "ribeirão-preto", "osasco", "sorocaba",
               "mauá", "são-josé-do-rio-preto", "mogi-das-cruzes", "santos", "diadema", "jundiaí",
               "piracicaba", "carapicuíba", "bauru", "itaquaquecetuba", "são-vicente", "franca",
               "praia-grande", "guarujá", "taubaté", "limeira", "suzano", "taboão-da-serra",
               "sumaré", "barueri", "embu-das-artes", "são-carlos", "indaiatuba", "cotia",
               "americana", "marília", "itapevi", "araraquara", "jacareí", "hortolândia",
               "presidente-prudente", "rio-claro", "araçatuba", "ferraz-de-vasconcelos",
               "santa-bárbara-d'oeste", "francisco-morato", "itapecerica-da-serra", "itu",
               "bragança-paulista", "pindamonhangaba", "itapetininga", "são-caetano-do-sul",
               "franco-da-rocha", "mogi-guaçu", "jaú", "botucatu", "atibaia",
               "santana-de-parnaíba", "araras", "cubatão", "valinhos", "sertãozinho", "jandira",
               "birigui", "ribeirão-pires", "votorantim", "barretos", "catanduva",
               "várzea-paulista", "guaratinguetá", "tatuí", "caraguatatuba", "itatiba", "salto",
               "poá", "ourinhos", "paulínia", "assis", "leme", "itanhaém", "caieiras",
               "mairiporã", "votuporanga", "itapeva", "caçapava", "mogi-mirim",
               "são-joão-da-boa-vista", "são-roque", "ubatuba", "avaré", "arujá", "são-sebastião",
               "lorena", "campo-limpo-paulista", "matão", "cruzeiro", "ibiúna", "vinhedo", "lins",
               "bebedouro", "jaboticabal", "cajamar", "pirassununga", "itapira", "cosmópolis",
               "amparo", "embu-guaçu", "fernandópolis", "mococa", "lençóis-paulista", "peruíbe",
               "tupã", "penápolis", "bertioga", "batatais", "itupeva", "boituva", "nova-odessa",
               "ibitinga", "mirassol", "monte-mor", "jaguariúna", "santa-isabel", "taquaritinga",
               "andradina", "mongaguá", "registro", "porto-ferreira", "capivari", "piedade",
               "são-josé-do-rio-pardo", "olímpia", "artur-nogueira", "porto-feliz",
               "vargem-grande-paulista", "campos-do-jordão", "são-joaquim-da-barra",
               "rio-grande-da-serra", "itararé", "monte-alto", "pontal", "cabreúva", "jales",
               "cerquilho", "louveira", "pedreira", "santa-cruz-do-rio-pardo", "tremembé",
               "mairinque", "capão-bonito", "dracena", "pederneiras", "paraguaçu-paulista",
               "salto-de-pirapora", "serrana", "garça", "jardinópolis",
               "espírito-santo-do-pinhal", "presidente-epitácio", "orlândia", "itápolis",
               "vargem-grande-do-sul", "tietê", "ituverava", "novo-horizonte", "socorro",
               "são-manuel", "guaíra", "américo-brasiliense", "promissão", "guariba",
               "pitangueiras", "presidente-venceslau", "agudos", "iperó", "josé-bonifácio",
               "aguaí", "aparecida", "barra-bonita", "são-pedro", "cravinhos", "bariri",
               "rio-das-pedras", "ibaté", "adamantina", "ilhabela", "santa-cruz-das-palmeiras",
               "araçoiaba-da-serra", "descalvado", "cachoeira-paulista", "morro-agudo",
               "guararapes", "são-miguel-arcanjo", "osvaldo-cruz", "barrinha", "biritiba-mirim",
               "santa-fé-do-sul", "juquitiba", "cândido-mota", "iguape", "igarapava",
               "casa-branca", "jarinu", "piraju", "guararema", "rancharia", "mirandópolis",
               "serra-negra", "pilar-do-sul", "cajati", "laranjal-paulista", "conchal",
               "santa-rita-do-passa-quatro", "pirapozinho", "dois-córregos", "piracaia", "itaí",
               "santa-gertrudes", "ilha-solteira", "santa-rosa-de-viterbo", "valparaíso",
               "martinópolis", "cajuru", "tanabi", "pereira-barreto", "pirajuí",
               "bom-jesus-dos-perdões", "angatuba", "monte-aprazível", "brodowski",
               "álvares-machado", "igaraçu-do-tietê", "potim", "cordeirópolis", "brotas", "apiaí",
               "iracemápolis", "santo-antônio-de-posse", "taquarituba", "tambaú",
               "teodoro-sampaio", "araçariguama", "miguelópolis", "palmital", "pompeia",
               "lucélia", "cunha", "nova-granada", "pradópolis", "guapiaçu", "guará", "castilho",
               "bastos", "santo-anastácio", "engenheiro-coelho", "capela-do-alto", "itatinga",
               "junqueirópolis", "regente-feijó", "paranapanema", "cerqueira-césar", "buri",
               "miracatu", "pariquera-açu", "monte-azul-paulista", "caconde", "viradouro",
               "pirapora-do-bom-jesus", "juquiá", "águas-de-lindoia", "alumínio",
               "nazaré-paulista", "guareí", "colina", "mirante-do-paranapanema", "paraibuna",
               "itirapina", "cesário-lange", "conchas", "jacupiranga", "elias-fausto",
               "cafelândia", "itaberá", "bady-bassitt", "severínia", "itariri", "potirendaba",
               "charqueada", "macatuba", "guapiara", "buritama", "salesópolis", "pindorama",
               "pedregulho", "rosana", "tabatinga", "ribeirão-branco", "ipuã", "altinópolis",
               "borborema", "fartura", "são-lourenço-da-serra", "panorama", "tupi-paulista",
               "eldorado", "santa-adélia", "são-simão", "itajobi", "pinhalzinho", "auriflama",
               "itaporanga", "tarumã", "ipaussu", "luiz-antônio", "holambra",
               "boa-esperança-do-sul", "santa-branca", "patrocínio-paulista", "serra-azul",
               "flórida-paulista", "pacaembu", "quatá", "maracaí", "itapuí", "taguaí", "urupês",
               "piquete", "avanhandava", "piratininga", "morungaba", "queluz", "valentim-gentil",
               "palmares-paulista", "joanópolis", "ribeirão-bonito", "presidente-bernardes",
               "tapiratiba", "palestina", "mineiros-do-tietê", "sete-barras", "cananeia",
               "riolândia", "duartina", "chavantes", "tabapuã", "ibirá", "bocaina", "cardoso",
               "são-sebastião-da-grama", "guaiçara", "lavínia", "sales-oliveira", "bofete",
               "iacanga", "nhandeara", "pirangi", "getulina", "pedro-de-toledo", "estiva-gerbi",
               "guaraci", "nova-europa", "ilha-comprida", "bernardino-de-campos", "divinolândia",
               "areiópolis", "parapuã", "bananal", "são-bento-do-sapucaí", "general-salgado",
               "vera-cruz", "rincão", "roseira", "são-luiz-do-paraitinga", "cajobi", "vargem",
               "ouroeste", "sarapuí", "uchoa", "torrinha", "rinópolis", "porangaba", "dumont",
               "manduri", "nova-campina", "ariranha", "reginópolis", "igaratá", "herculândia",
               "euclides-da-cunha-paulista", "terra-roxa", "salto-grande", "palmeira-d'oeste",
               "iaras", "cedral", "urânia", "rafard", "bálsamo", "paulo-de-faria",
               "neves-paulista", "dobrada", "são-josé-da-bela-vista", "dourado", "santa-lúcia",
               "vista-alegre-do-alto", "pereiras", "cristais-paulista", "clementina",
               "ouro-verde", "arealva", "santo-antônio-do-aracanguá", "estrela-d'oeste",
               "guaraçaí", "irapuru", "saltinho", "icém", "águas-da-prata", "iepê", "macaubal",
               "monte-alegre-do-sul", "bilac", "irapuã", "lindoia", "itobi", "tapiraí", "catiguá",
               "ibirarema", "sud-mennucci", "ribeirão-grande", "são-pedro-do-turvo",
               "barra-do-turvo", "guatapará", "restinga", "ipeúna", "tarabai", "nuporanga",
               "barbosa", "pauliceia", "cosmorama", "lavrinhas", "jaci", "orindiúva",
               "nova-aliança", "jaborandi", "santo-antônio-da-alegria", "tuiuti",
               "santo-antônio-do-pinhal", "anhembi", "guarantã", "natividade-da-serra",
               "quintana", "jambeiro", "gália", "oriente", "itirapuã", "paraíso", "pardinho",
               "arandu", "sales", "iacri", "ubarana", "silveiras", "taiaçu", "taciba", "colômbia",
               "santa-maria-da-serra", "echaporã", "pedra-bela", "águas-de-santa-bárbara",
               "poloni", "coroados", "alambari", "campina-do-monte-alegre", "santa-albertina",
               "piacatu", "américo-de-campos", "santo-antônio-do-jardim", "caiuá",
               "marabá-paulista", "taquarivaí", "novais", "três-fronteiras", "luiziânia",
               "fernando-prestes", "guaimbê", "balbinos", "barra-do-chapéu", "braúna", "aramina",
               "santa-ernestina", "sabino", "taiúva", "riversul", "mendonça", "avaí", "ipiguá",
               "salmourão", "guzolândia", "pratânia", "planalto", "álvaro-de-carvalho", "canitar",
               "nipoã", "canas", "analândia", "campos-novos-paulista", "itapura", "lagoinha",
               "indiana", "mirassolândia", "narandiba", "espírito-santo-do-turvo", "boraceia",
               "glicério", "gastão-vidigal", "gavião-peixoto", "ubirajara",
               "santópolis-do-aguapeí", "júlio-mesquita", "motuca", "ribeirão-corrente",
               "coronel-macedo", "monteiro-lobato", "pontalinda", "lupércio", "ribeirão-do-sul",
               "tejupá", "joão-ramalho", "santa-cruz-da-conceição", "murutinga-do-sul",
               "buritizal", "onda-verde", "sandovalina", "ocauçu", "cabrália-paulista",
               "itapirapuã-paulista", "iporanga", "aparecida-d'oeste", "caiabu", "populina",
               "alfredo-marcondes", "monte-castelo", "altair", "são-josé-do-barreiro", "anhumas",
               "nova-luzitânia", "alto-alegre", "presidente-alves", "pracinha", "mariápolis",
               "paranapuã", "corumbataí", "inúbia-paulista", "nova-independência", "suzanápolis",
               "bom-sucesso-de-itararé", "indiaporã", "areias", "redenção-da-serra", "meridiano",
               "itaju", "quadra", "macedônia", "piquerobi", "álvares-florence", "elisiário",
               "sarutaiá", "rifaina", "adolfo", "platina", "sebastianópolis-do-sul", "mombuca",
               "barão-de-antonina", "águas-de-são-pedro", "pongaí", "queiroz", "jumirim",
               "ribeira", "itaoca", "alvinlândia", "emilianópolis", "jeriquara", "rubineia",
               "nantes", "rubiácea", "magda", "santo-expedito", "pedrinhas-paulista",
               "mira-estrela", "marapoama", "bento-de-abreu", "santa-mercedes", "floreal",
               "brejo-alegre", "são-francisco", "taquaral", "cândido-rodrigues",
               "gabriel-monteiro", "estrela-do-norte", "zacarias", "florínea", "timburi",
               "borebi", "lutécia", "oscar-bressane", "pontes-gestal", "são-joão-das-duas-pontes",
               "cássia-dos-coqueiros", "santa-rita-d'oeste", "óleo", "pedranópolis", "arapeí",
               "embaúba", "sagres", "torre-de-pedra", "lucianópolis", "nova-guataporanga",
               "lourdes", "monções", "ribeirão-dos-índios", "parisi", "santa-cruz-da-esperança",
               "dolcinópolis", "santa-clara-d'oeste", "marinópolis", "são-joão-do-pau-d'alho",
               "cruzália", "turiúba", "guarani-d'oeste", "são-joão-de-iracema", "mesópolis",
               "nova-canaã-paulista", "união-paulista", "vitória-brasil", "paulistânia",
               "aspásia", "dirce-reis", "arco-íris", "turmalina", "trabiju", "fernão",
               "santa-salete", "santana-da-ponte-pensa", "flora-rica", "nova-castilho", "uru",
               "borá", "são_paulo", "guarulhos", "campinas", "são_bernardo_do_campo",
               "são_josé_dos_campos", "santo_andré", "ribeirão_preto", "osasco", "sorocaba",
               "mauá", "são_josé_do_rio_preto", "mogi_das_cruzes", "santos", "diadema", "jundiaí",
               "piracicaba", "carapicuíba", "bauru", "itaquaquecetuba", "são_vicente", "franca",
               "praia_grande", "guarujá", "taubaté", "limeira", "suzano", "taboão_da_serra",
               "sumaré", "barueri", "embu_das_artes", "são_carlos", "indaiatuba", "cotia",
               "americana", "marília", "itapevi", "araraquara", "jacareí", "hortolândia",
               "presidente_prudente", "rio_claro", "araçatuba", "ferraz_de_vasconcelos",
               "santa_bárbara_d'oeste", "francisco_morato", "itapecerica_da_serra", "itu",
               "bragança_paulista", "pindamonhangaba", "itapetininga", "são_caetano_do_sul",
               "franco_da_rocha", "mogi_guaçu", "jaú", "botucatu", "atibaia",
               "santana_de_parnaíba", "araras", "cubatão", "valinhos", "sertãozinho", "jandira",
               "birigui", "ribeirão_pires", "votorantim", "barretos", "catanduva",
               "várzea_paulista", "guaratinguetá", "tatuí", "caraguatatuba", "itatiba", "salto",
               "poá", "ourinhos", "paulínia", "assis", "leme", "itanhaém", "caieiras",
               "mairiporã", "votuporanga", "itapeva", "caçapava", "mogi_mirim",
               "são_joão_da_boa_vista", "são_roque", "ubatuba", "avaré", "arujá", "são_sebastião",
               "lorena", "campo_limpo_paulista", "matão", "cruzeiro", "ibiúna", "vinhedo", "lins",
               "bebedouro", "jaboticabal", "cajamar", "pirassununga", "itapira", "cosmópolis",
               "amparo", "embu-guaçu", "fernandópolis", "mococa", "lençóis_paulista", "peruíbe",
               "tupã", "penápolis", "bertioga", "batatais", "itupeva", "boituva", "nova_odessa",
               "ibitinga", "mirassol", "monte_mor", "jaguariúna", "santa_isabel", "taquaritinga",
               "andradina", "mongaguá", "registro", "porto_ferreira", "capivari", "piedade",
               "são_josé_do_rio_pardo", "olímpia", "artur_nogueira", "porto_feliz",
               "vargem_grande_paulista", "campos_do_jordão", "são_joaquim_da_barra",
               "rio_grande_da_serra", "itararé", "monte_alto", "pontal", "cabreúva", "jales",
               "cerquilho", "louveira", "pedreira", "santa_cruz_do_rio_pardo", "tremembé",
               "mairinque", "capão_bonito", "dracena", "pederneiras", "paraguaçu_paulista",
               "salto_de_pirapora", "serrana", "garça", "jardinópolis",
               "espírito_santo_do_pinhal", "presidente_epitácio", "orlândia", "itápolis",
               "vargem_grande_do_sul", "tietê", "ituverava", "novo_horizonte", "socorro",
               "são_manuel", "guaíra", "américo_brasiliense", "promissão", "guariba",
               "pitangueiras", "presidente_venceslau", "agudos", "iperó", "josé_bonifácio",
               "aguaí", "aparecida", "barra_bonita", "são_pedro", "cravinhos", "bariri",
               "rio_das_pedras", "ibaté", "adamantina", "ilhabela", "santa_cruz_das_palmeiras",
               "araçoiaba_da_serra", "descalvado", "cachoeira_paulista", "morro_agudo",
               "guararapes", "são_miguel_arcanjo", "osvaldo_cruz", "barrinha", "biritiba_mirim",
               "santa_fé_do_sul", "juquitiba", "cândido_mota", "iguape", "igarapava",
               "casa_branca", "jarinu", "piraju", "guararema", "rancharia", "mirandópolis",
               "serra_negra", "pilar_do_sul", "cajati", "laranjal_paulista", "conchal",
               "santa_rita_do_passa_quatro", "pirapozinho", "dois_córregos", "piracaia", "itaí",
               "santa_gertrudes", "ilha_solteira", "santa_rosa_de_viterbo", "valparaíso",
               "martinópolis", "cajuru", "tanabi", "pereira_barreto", "pirajuí",
               "bom_jesus_dos_perdões", "angatuba", "monte_aprazível", "brodowski",
               "álvares_machado", "igaraçu_do_tietê", "potim", "cordeirópolis", "brotas", "apiaí",
               "iracemápolis", "santo_antônio_de_posse", "taquarituba", "tambaú",
               "teodoro_sampaio", "araçariguama", "miguelópolis", "palmital", "pompeia",
               "lucélia", "cunha", "nova_granada", "pradópolis", "guapiaçu", "guará", "castilho",
               "bastos", "santo_anastácio", "engenheiro_coelho", "capela_do_alto", "itatinga",
               "junqueirópolis", "regente_feijó", "paranapanema", "cerqueira_césar", "buri",
               "miracatu", "pariquera-açu", "monte_azul_paulista", "caconde", "viradouro",
               "pirapora_do_bom_jesus", "juquiá", "águas_de_lindoia", "alumínio",
               "nazaré_paulista", "guareí", "colina", "mirante_do_paranapanema", "paraibuna",
               "itirapina", "cesário_lange", "conchas", "jacupiranga", "elias_fausto",
               "cafelândia", "itaberá", "bady_bassitt", "severínia", "itariri", "potirendaba",
               "charqueada", "macatuba", "guapiara", "buritama", "salesópolis", "pindorama",
               "pedregulho", "rosana", "tabatinga", "ribeirão_branco", "ipuã", "altinópolis",
               "borborema", "fartura", "são_lourenço_da_serra", "panorama", "tupi_paulista",
               "eldorado", "santa_adélia", "são_simão", "itajobi", "pinhalzinho", "auriflama",
               "itaporanga", "tarumã", "ipaussu", "luiz_antônio", "holambra",
               "boa_esperança_do_sul", "santa_branca", "patrocínio_paulista", "serra_azul",
               "flórida_paulista", "pacaembu", "quatá", "maracaí", "itapuí", "taguaí", "urupês",
               "piquete", "avanhandava", "piratininga", "morungaba", "queluz", "valentim_gentil",
               "palmares_paulista", "joanópolis", "ribeirão_bonito", "presidente_bernardes",
               "tapiratiba", "palestina", "mineiros_do_tietê", "sete_barras", "cananeia",
               "riolândia", "duartina", "chavantes", "tabapuã", "ibirá", "bocaina", "cardoso",
               "são_sebastião_da_grama", "guaiçara", "lavínia", "sales_oliveira", "bofete",
               "iacanga", "nhandeara", "pirangi", "getulina", "pedro_de_toledo", "estiva_gerbi",
               "guaraci", "nova_europa", "ilha_comprida", "bernardino_de_campos", "divinolândia",
               "areiópolis", "parapuã", "bananal", "são_bento_do_sapucaí", "general_salgado",
               "vera_cruz", "rincão", "roseira", "são_luiz_do_paraitinga", "cajobi", "vargem",
               "ouroeste", "sarapuí", "uchoa", "torrinha", "rinópolis", "porangaba", "dumont",
               "manduri", "nova_campina", "ariranha", "reginópolis", "igaratá", "herculândia",
               "euclides_da_cunha_paulista", "terra_roxa", "salto_grande", "palmeira_d'oeste",
               "iaras", "cedral", "urânia", "rafard", "bálsamo", "paulo_de_faria",
               "neves_paulista", "dobrada", "são_josé_da_bela_vista", "dourado", "santa_lúcia",
               "vista_alegre_do_alto", "pereiras", "cristais_paulista", "clementina",
               "ouro_verde", "arealva", "santo_antônio_do_aracanguá", "estrela_d'oeste",
               "guaraçaí", "irapuru", "saltinho", "icém", "águas_da_prata", "iepê", "macaubal",
               "monte_alegre_do_sul", "bilac", "irapuã", "lindoia", "itobi", "tapiraí", "catiguá",
               "ibirarema", "sud_mennucci", "ribeirão_grande", "são_pedro_do_turvo",
               "barra_do_turvo", "guatapará", "restinga", "ipeúna", "tarabai", "nuporanga",
               "barbosa", "pauliceia", "cosmorama", "lavrinhas", "jaci", "orindiúva",
               "nova_aliança", "jaborandi", "santo_antônio_da_alegria", "tuiuti",
               "santo_antônio_do_pinhal", "anhembi", "guarantã", "natividade_da_serra",
               "quintana", "jambeiro", "gália", "oriente", "itirapuã", "paraíso", "pardinho",
               "arandu", "sales", "iacri", "ubarana", "silveiras", "taiaçu", "taciba", "colômbia",
               "santa_maria_da_serra", "echaporã", "pedra_bela", "águas_de_santa_bárbara",
               "poloni", "coroados", "alambari", "campina_do_monte_alegre", "santa_albertina",
               "piacatu", "américo_de_campos", "santo_antônio_do_jardim", "caiuá",
               "marabá_paulista", "taquarivaí", "novais", "três_fronteiras", "luiziânia",
               "fernando_prestes", "guaimbê", "balbinos", "barra_do_chapéu", "braúna", "aramina",
               "santa_ernestina", "sabino", "taiúva", "riversul", "mendonça", "avaí", "ipiguá",
               "salmourão", "guzolândia", "pratânia", "planalto", "álvaro_de_carvalho", "canitar",
               "nipoã", "canas", "analândia", "campos_novos_paulista", "itapura", "lagoinha",
               "indiana", "mirassolândia", "narandiba", "espírito_santo_do_turvo", "boraceia",
               "glicério", "gastão_vidigal", "gavião_peixoto", "ubirajara",
               "santópolis_do_aguapeí", "júlio_mesquita", "motuca", "ribeirão_corrente",
               "coronel_macedo", "monteiro_lobato", "pontalinda", "lupércio", "ribeirão_do_sul",
               "tejupá", "joão_ramalho", "santa_cruz_da_conceição", "murutinga_do_sul",
               "buritizal", "onda_verde", "sandovalina", "ocauçu", "cabrália_paulista",
               "itapirapuã_paulista", "iporanga", "aparecida_d'oeste", "caiabu", "populina",
               "alfredo_marcondes", "monte_castelo", "altair", "são_josé_do_barreiro", "anhumas",
               "nova_luzitânia", "alto_alegre", "presidente_alves", "pracinha", "mariápolis",
               "paranapuã", "corumbataí", "inúbia_paulista", "nova_independência", "suzanápolis",
               "bom_sucesso_de_itararé", "indiaporã", "areias", "redenção_da_serra", "meridiano",
               "itaju", "quadra", "macedônia", "piquerobi", "álvares_florence", "elisiário",
               "sarutaiá", "rifaina", "adolfo", "platina", "sebastianópolis_do_sul", "mombuca",
               "barão_de_antonina", "águas_de_são_pedro", "pongaí", "queiroz", "jumirim",
               "ribeira", "itaoca", "alvinlândia", "emilianópolis", "jeriquara", "rubineia",
               "nantes", "rubiácea", "magda", "santo_expedito", "pedrinhas_paulista",
               "mira_estrela", "marapoama", "bento_de_abreu", "santa_mercedes", "floreal",
               "brejo_alegre", "são_francisco", "taquaral", "cândido_rodrigues",
               "gabriel_monteiro", "estrela_do_norte", "zacarias", "florínea", "timburi",
               "borebi", "lutécia", "oscar_bressane", "pontes_gestal", "são_joão_das_duas_pontes",
               "cássia_dos_coqueiros", "santa_rita_d'oeste", "óleo", "pedranópolis", "arapeí",
               "embaúba", "sagres", "torre_de_pedra", "lucianópolis", "nova_guataporanga",
               "lourdes", "monções", "ribeirão_dos_índios", "parisi", "santa_cruz_da_esperança",
               "dolcinópolis", "santa_clara_d'oeste", "marinópolis", "são_joão_do_pau_d'alho",
               "cruzália", "turiúba", "guarani_d'oeste", "são_joão_de_iracema", "mesópolis",
               "nova_canaã_paulista", "união_paulista", "vitória_brasil", "paulistânia",
               "aspásia", "dirce_reis", "arco-íris", "turmalina", "trabiju", "fernão",
               "santa_salete", "santana_da_ponte_pensa", "flora_rica", "nova_castilho", "uru",
               "borá", "acre", "alagoas", "amapá", "amazonas", "bahia", "ceará",
               "distrito_federal", "espírito_santo", "goiás", "maranhão", "mato_grosso",
               "mato_grosso_do_sul", "minas_gerais", "pará", "paraíba", "paraná", "pernambuco",
               "piauí", "rio_de_janeiro", "rio_grande_do_norte", "rio_grande_do_sul", "rondônia",
               "roraima", "santa_catarina", "são_paulo", "sergipe", "tocantins", "rio_branco",
               "maceió", "macapá", "manaus", "salvador", "fortaleza", "brasília", "vitória",
               "goiânia", "são_luís", "cuiabá", "campo_grande", "belo_horizonte", "belém",
               "joão_pessoa", "curitiba", "recife", "teresina", "rio_de_janeiro", "natal",
               "porto_alegre", "porto_velho", "boa_vista", "florianópolis", "são_paulo",
               "aracaju", "palmas", "acre", "alagoas", "amapá", "amazonas", "bahia", "ceará",
               "distrito-federal", "espírito-santo", "goiás", "maranhão", "mato-grosso",
               "mato-grosso-do-sul", "minas-gerais", "pará", "paraíba", "paraná", "pernambuco",
               "piauí", "rio-de-janeiro", "rio-grande-do-norte", "rio-grande-do-sul", "rondônia",
               "roraima", "santa-catarina", "são-paulo", "sergipe", "tocantins", "rio-branco",
               "maceió", "macapá", "manaus", "salvador", "fortaleza", "brasília", "vitória",
               "goiânia", "são-luís", "cuiabá", "campo-grande", "belo-horizonte", "belém",
               "joão-pessoa", "curitiba", "recife", "teresina", "rio-de-janeiro", "natal",
               "porto-alegre", "porto-velho", "boa-vista", "florianópolis", "são-paulo",
               "aracaju", "palmas"]
TYPOS = ['', '"', 'eo', 'ea', 'eos', 'au', 'eas', 'ede', 'eda', 'edo', 'edu', 'edocumentos,', 'acerca', 'ad', 'cara',
         'car', 'f', 'g', 'h', 'in', 'j', 'n', 's', 's/a']
SELECTED_PROPER_NAMES = ['adélia', 'andré', 'bárbara', 'barbara', 'barreto', 'bento', 'bernardo', 'bragança', 'caetano',
                         'cândido', 'carlos', 'carvalho', 'cassia', 'catarina', 'conceição', 'dirce', 'embu', 'guaçu',
                         'guaçú', 'isabel', 'itapecerica', 'itaquera', 'joão', 'joaquim', 'jordão', 'maria', 'mateus',
                         'mogi', 'moji', 'monteiro', 'moóca', 'mooca', 'moreira', 'nogueira', 'palmares', 'paraguaçu',
                         'paulo', 'paulinia', 'padrinhas', 'pedro', 'penha', 'rita', 'roque', 'sapucaí', 'sebastião',
                         'simão', 'vicente']
STOPWORDS = nltk.corpus.stopwords.words(
    'portuguese') + PLACE_NAMES + TYPOS + SELECTED_PROPER_NAMES


TO_REMOVE_FIRST_CHAR = ['eobjetiva', 'eemolumentos', 'eanexos', 'àexecução', 'aadministração', 'eimpenhorabilidade',
                        'eambiental', 'aexecução', 'eincomunicabilidade', 'eautenticação', 'eapreensão', 'eauxiliares',
                        'eespecificação', 'aalienacao', 'eóbito', 'aoutorga', 'evenda', 'edocumentos', 'etabelião',
                        'eregistrais', 'ecompra', 'equalitativa', 'equantitativa', 'etítulos', 'etabelionato',
                        'eperpetuidade', 'etributário', 'econtribuições', 'eregistros', 'eratificação', 'eregistral',
                        'ecálculo', 'ereceita', 'eregistro', 'enotarial', 'econfrontação', 'edeterminada', 'esubjetiva',
                        'enotas', 'etransferência', 'econfrontações', 'epé', 'etutelas', 'ecivil', 'eregistradores',
                        'ecpf', 'eprotesto', 'eprocurações', 'eprática', 'econvenção', 'enão', 'efiscal', 'ehipotecária',
                        'ecessão', 'epartilha', 'edistribuição', 'edespesa', 'epessoais', 'ecorreições', 'efinanceira',
                        'epraça', 'apreliminar', 'opedido', 'operíodo', 'aterceiro', 'aredação', 'aresolução', 'alei',
                        'àlavratura', 'àalienação', 'aalienação', 'acontagem', 'àdelegação', 'adelegação', 'adelegados',
                        'àlei', 'amaior', 'àmaior', 'aregistro', 'áregistro', 'aterceiros', 'àterceiro', 'àterceiros',
                        'eárea', 'earea', 'eáreas', 'ecertidão', 'ecertidões', 'ecerto', 'etítulo', 'etitulos', 'étitular',
                        'eurbanístico', 'eurbanística']
TO_REMOVE_LAST_CHAR = ['imóveis,', 'paulista,', 'ad_mensuram"']

SPECIAL_STEM_DICT = {'acórdão': 'acord__', 'acórdãos': 'acord__', 'analógica': 'analog__', 'apelido': 'apel__',
                     'assento': 'assent__', 'assentos': 'assent__', 'áudio': 'audi__', 'autor': 'autor__',
                     'autoria': 'autor__', 'bancário': 'banc__', 'carta': 'cart__', 'cartas': 'cart__',
                     'carteira': 'cart___', 'casamento': 'cas__', 'casada': 'cas__', 'casamentos': 'cas__',
                     'casado': 'cas__', 'casados': 'cas__', 'caso': 'cas___', 'casos': 'cas___', 'certidão': 'cert__',
                     'certidões': 'cert__', 'cessação': 'cess__', 'cessar': 'cess__', 'cessada': 'cess__',
                     'cessantes': 'cess__', 'cidade': 'cidad__', 'comprida': 'compr__', 'consultivo': 'consul__',
                     'consumado': 'consum__', 'contado': 'cont', 'contração': 'contr', 'contratantes': 'contrat__',
                     'contratação': 'contrat__', 'contratado': 'contrat__', 'correição': 'corre__',
                     'correições': 'corre__', 'corretivos': 'corre__', 'corregedor': 'correg__',
                     'corregedores': 'correg__', 'diretoria': 'diret__', 'dólar': 'dol__', 'equipe': 'equip__',
                     'estrição': 'estr__', 'extração': 'extr__', 'fila': 'fil__', 'funda': 'fund__', 'fundo': 'fund__',
                     'garagem': 'gar__', 'general': 'gener__', 'gerência': 'ger__', 'ilegal': 'ileg__',
                     'imposta': 'impost__', 'incrição': 'incr__', 'informalismo': 'inform__', 'informal': 'inform__',
                     'inter': 'int__', 'íntimo': 'intim__', 'intimidade': 'intim__', 'jornadas': 'jorn__',
                     'jornada': 'jorn__', 'juntada': 'junt__', 'juros': 'jur', 'liquidez': 'liquid__',
                     'liquidante': 'liquid__', 'livro': 'livr__', 'livros': 'livr__', 'maio': 'mai__',
                     'mandatário': 'mandat__', 'marítimas': 'marit__', 'minas': 'min__', 'mutação': 'mut__',
                     'palmo': 'palm', 'parcial': 'parc__', 'parto': 'part__', 'partos': 'part__', 'plantão': 'plant__',
                     'pontos': 'pont__', 'ponto': 'pont__', 'portaria': 'port__', 'portarias': 'port__',
                     'portal': 'port__', 'porto': 'port___', 'posturas': 'post__', 'precário': 'prec__',
                     'pretações': 'pret__', 'procedimento': 'proced__', 'procedimentos': 'proced__', 'provas': 'prov__',
                     'prova': 'prov__', 'prov': 'prov__', 'público': 'public__', 'pública': 'public__',
                     'públicas': 'public__', 'públicos': 'public__', 'redondo': 'red__', 'rede': 'red__',
                     'região': 'reg__', 'regulamento': 'regul__', 'relato': 'relat__', 'relevo': 'relev__',
                     'residual': 'resid__', 'sábados': 'sab__', 'sete': 'set__', 'sindicância': 'sindic__',
                     'sindicâncias': 'sindic__', 'sócia': 'soc__', 'solo': 'sol__', 'solteira': 'solt__',
                     'solteiro': 'solt__', 'tabela': 'tabel__', 'tabelas': 'tabel__', 'terraço': 'terr__',
                     'testador': 'test__', 'teste': 'test__', 'universidade': 'univers__', 'vais': 'val__',
                     'válida': 'val__', 'válido': 'val__', 'varão': 'var__', 'verba': 'verb__', 'verbas': 'verb__',
                     'vidual': 'vid__', 'vigário': 'vig__', 'vigor': 'vig___', 'vila': 'vil__', 'acesso': 'acess__'}

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
                and ('-' in verbete or '–' in verbete or '_' in verbete)\
                and not special_split:
            splitted_class = re.split(
                '(?<!nº)(?<!cg)(?<!\d)[-–_]+|[-–_]+(?=[0-9]+[ºª°]{1})|(?<=\d)[-–_]+(?=\D)', verbete)
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

        elif special_split and ('-' in verbete or '–' in verbete or '_' in verbete):
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
                splitted_class = re.split('(?<!nº)(?<!cg)(?<!\d)[-–_]+|[-–_]+(?=[0-9]+[ºª°]{1})|(?<=\d)[-–_]+(?=\D)',
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

                splitted_class = re.split('(?<!nº)(?<!cg)(?<!\d)[-–_]+|[-–_]+(?=[0-9]+[ºª°]{1})|(?<=\d)[-–_]+(?=\D)',
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
              ' e redução prévia para ' + str(nc_red) + ' dim')
    if nc_red == 0:
        plt.title('Clustering com ' + CLUSTERING_METHOD +
                  'sem redução dimensional prévia')
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
            print('Alerta: classe de nivel 0 não correspondente a um cluster.')
        lvl0_verbs.append(verb.ancestors[1].name)
    lvl0_verbs = np.unique(lvl0_verbs).tolist()
    processo.verbetacao = lvl0_verbs

# Save this version to disk
with open('koll_corpus_KM50_lvl0.pkl', 'wb') as f:
    pickle.dump(corpus_lvl0only, f)
with open('koll_fields_KM50_lvl0.pkl', 'wb') as f:
    pickle.dump(fileds, f)
