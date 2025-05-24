import pandas as pd
from wordcloud import WordCloud
from konlpy.tag import Okt
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_excel('통합 문서3.xlsx')
texts = df['review'].dropna().astype(str).tolist()

# 2. 형태소 분석 및 명사 추출
okt = Okt()
nouns = []
for text in texts:
    nouns += okt.nouns(text)

# 3. 불용어(원하지 않는 단어) 제거 (예시)
stopwords = set(['버스', '정류장', '기사', '시간', '앱', '어플', '이용', '사용', '정도', '때문', '정말', '너무'])
words = [word for word in nouns if word not in stopwords and len(word) > 1]

# 4. 단어 빈도수 집계
from collections import Counter
word_counts = Counter(words)

# 5. 워드클라우드 생성
wc = WordCloud(font_path='/System/Library/Fonts/AppleGothic.ttf', background_color='white', width=800, height=600)
cloud = wc.generate_from_frequencies(word_counts)

plt.rcParams['font.family'] = 'AppleGothic'  # 맥OS 기준
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 8))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.title("리뷰 주요 키워드 워드클라우드")
plt.show()

cloud.to_file('wordcloud_result.png')