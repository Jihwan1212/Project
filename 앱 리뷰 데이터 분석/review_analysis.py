import pandas as pd
import numpy as np
from google_play_scraper import Sort, reviews
import urllib.request
import json
from datetime import datetime
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def get_google_play_reviews(app_id, lang='ko', country='kr', count=None):
    reviews_list = []
    continuation_token = None
    
    while True:
        result, continuation_token = reviews(
            app_id,
            lang=lang,
            country=country,
            count=100,  # Maximum count per request
            continuation_token=continuation_token,
            sort=Sort.NEWEST
        )
        
        if not result:
            break
            
        reviews_list.extend(result)
        print(f"Collected {len(reviews_list)} reviews from Google Play Store")
        
        if not continuation_token or (count and len(reviews_list) >= count):
            break
    
    df = pd.DataFrame(reviews_list)
    df['platform'] = 'Google Play'
    df['date'] = pd.to_datetime(df['at']).dt.tz_localize(None)
    df = df.rename(columns={'content': 'review', 'score': 'rating'})
    
    if count:
        df = df.head(count)
    
    return df[['review', 'rating', 'date', 'platform']]

def get_app_store_reviews(app_id, country='kr', count=None):
    all_reviews = []
    page = 1
    
    while True:
        url = f'https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostRecent/page={page}/json'
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                entries = data['feed'].get('entry', [])
                
                # Check if we got any reviews
                if not entries or (isinstance(entries, dict) and 'author' in entries):
                    # If entries is a dict, it means we got feed metadata instead of reviews
                    break
                    
                for entry in entries:
                    if isinstance(entry, dict) and 'content' in entry:
                        review = {
                            'review': entry['content']['label'],
                            'rating': int(entry['im:rating']['label']),
                            'date': pd.to_datetime(entry['updated']['label']).tz_localize(None),
                            'platform': 'App Store'
                        }
                        all_reviews.append(review)
                
                print(f"Collected {len(all_reviews)} reviews from App Store")
                
                if count and len(all_reviews) >= count:
                    break
                    
                page += 1
                
        except (KeyError, TypeError, json.JSONDecodeError, urllib.error.URLError) as e:
            print(f"Warning: Could not parse App Store reviews on page {page}: {e}")
            break
    
    df = pd.DataFrame(all_reviews)
    
    if count:
        df = df.head(count)
    
    return df

def analyze_sentiment(df):
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="beomi/KcELECTRA-base-v2022",
        tokenizer="beomi/KcELECTRA-base-v2022"
    )
    
    df['sentiment_score'] = 0.0
    df['sentiment'] = ''
    df['model_sentiment'] = ''
    
    total = len(df)
    for idx, row in df.iterrows():
        try:
            # 리뷰 텍스트가 비어있거나 너무 짧은 경우 평점만으로 판단
            if pd.isna(row['review']) or len(str(row['review']).strip()) < 5:
                df.at[idx, 'sentiment_score'] = 0.5 if row['rating'] == 3 else (0.9 if row['rating'] > 3 else 0.1)
                df.at[idx, 'model_sentiment'] = 'neutral'
                df.at[idx, 'sentiment'] = 'LABEL_1' if row['rating'] > 3 else ('LABEL_0' if row['rating'] < 3 else 'neutral')
            else:
                result = sentiment_analyzer(row['review'])[0]
                df.at[idx, 'model_sentiment'] = result['label']
                model_score = result['score']
                
                # 평점과 감성 분석 결과를 결합
                rating_score = (row['rating'] - 1) / 4  # 1-5 점수를 0-1로 정규화
                combined_score = (model_score + rating_score) / 2
                df.at[idx, 'sentiment_score'] = combined_score
                
                # 최종 감성 레이블 결정
                if row['rating'] >= 4:  # 높은 평점
                    df.at[idx, 'sentiment'] = 'LABEL_1'
                elif row['rating'] <= 2:  # 낮은 평점
                    df.at[idx, 'sentiment'] = 'LABEL_0'
                elif row['rating'] == 3:  # 중립 평점
                    df.at[idx, 'sentiment'] = 'neutral'
                else:  # 평점이 3점대인 경우 감성 분석 결과를 따름
                    df.at[idx, 'sentiment'] = result['label']
            
            if (idx + 1) % 10 == 0:
                print(f"Analyzed {idx + 1}/{total} reviews")
                
        except Exception as e:
            print(f"Warning: Could not analyze sentiment for review at index {idx}")
            df.at[idx, 'sentiment_score'] = 0.5
            df.at[idx, 'model_sentiment'] = 'neutral'
            df.at[idx, 'sentiment'] = 'neutral'
    
    return df

def visualize_results(df):
    plt.figure(figsize=(20, 15))
    
    # Rating distribution
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='rating', hue='platform')
    plt.title('Rating Distribution by Platform')
    
    # Sentiment distribution
    plt.subplot(2, 2, 2)
    sns.countplot(data=df, x='sentiment', hue='platform')
    plt.title('Sentiment Distribution by Platform')
    
    # Average rating over time
    plt.subplot(2, 2, 3)
    df_grouped = df.groupby(['platform', pd.Grouper(key='date', freq='ME')])['rating'].mean().reset_index()
    sns.lineplot(data=df_grouped, x='date', y='rating', hue='platform')
    plt.title('Average Rating Over Time')
    
    # Sentiment score distribution
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='platform', y='sentiment_score')
    plt.title('Sentiment Score Distribution by Platform')
    
    plt.tight_layout()
    plt.savefig('review_analysis_results.png')
    plt.close()

def main():
    # App IDs
    GOOGLE_PLAY_APP_ID = 'com.gscaltex.energyplus'
    APP_STORE_APP_ID = '1538919072'
    
    print("Collecting reviews from Google Play Store...")
    google_play_df = get_google_play_reviews(GOOGLE_PLAY_APP_ID)
    
    print("\nCollecting reviews from App Store...")
    app_store_df = get_app_store_reviews(APP_STORE_APP_ID)
    
    # Combine reviews
    df = pd.concat([google_play_df, app_store_df], ignore_index=True)
    
    print(f"\nTotal reviews collected: {len(df)}")
    print(f"Google Play Store reviews: {len(google_play_df)}")
    print(f"App Store reviews: {len(app_store_df)}")
    
    print("\nAnalyzing sentiment...")
    df = analyze_sentiment(df)
    
    # Save results
    df.to_csv('review_analysis.csv', index=False, encoding='utf-8-sig')
    
    # Visualize results
    visualize_results(df)
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total reviews analyzed: {len(df)}")
    print("\nAverage rating by platform:")
    print(df.groupby('platform')['rating'].mean())
    print("\nSentiment distribution by platform:")
    sentiment_dist = df.groupby(['platform', 'sentiment']).size().unstack(fill_value=0)
    print(sentiment_dist)
    
    # 평점별 감성 분포
    print("\nSentiment distribution by rating:")
    rating_sentiment = pd.crosstab(df['rating'], df['sentiment'])
    print(rating_sentiment)
    
    # 모델의 원래 감성 분석 결과와 최종 결과 비교
    print("\nModel vs Final Sentiment Distribution:")
    model_vs_final = pd.crosstab(df['model_sentiment'], df['sentiment'])
    print(model_vs_final)

if __name__ == "__main__":
    main()
