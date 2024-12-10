import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv('word_citedby_predict.csv')

from wordcloud import WordCloud

# สร้าง Word Cloud
word_dict = dict(zip(df['word'], df['citedby_predict']))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_dict)

# แสดง Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Predicted Citedby Contributions')
plt.show()