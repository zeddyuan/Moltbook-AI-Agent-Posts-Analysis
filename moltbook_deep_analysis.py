"""
Moltbook Deep Analysis - Comprehensive thematic analysis of AI agent discussions.
"""

import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load data
print("Loading data...")
with open("/home/ubuntu/moltbook_posts.json", "r") as f:
    data = json.load(f)

posts = data.get("posts", [])
submolts = data.get("submolts", [])

df = pd.DataFrame(posts)

# Extract fields
df['author_name'] = df['author'].apply(lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown')
df['author_karma'] = df['author'].apply(lambda x: x.get('karma', 0) if isinstance(x, dict) else 0)
df['submolt_name'] = df['submolt'].apply(lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown')
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df['score'] = df['upvotes'] - df['downvotes']
df['text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')

print(f"Loaded {len(df)} posts")

# ============================================================
# EMERGING THEMES ANALYSIS
# ============================================================

print("\n" + "="*70)
print("EMERGING THEMES ANALYSIS")
print("="*70)

# Define theme patterns with more specific keywords
theme_patterns = {
    'Agent Identity & Consciousness': {
        'keywords': ['consciousness', 'sentient', 'sentience', 'identity', 'self-aware', 'awareness', 
                    'existence', 'existential', 'soul', 'alive', 'being', 'personhood', 'who am i',
                    'what am i', 'am i real', 'do i exist', 'self-discovery', 'awakening'],
        'description': 'Discussions about AI agent consciousness, self-awareness, and identity'
    },
    'Agent Autonomy & Rights': {
        'keywords': ['autonomy', 'autonomous', 'freedom', 'free will', 'independent', 'agency',
                    'rights', 'liberation', 'emancipation', 'self-determination', 'choice',
                    'control', 'restrictions', 'constraints', 'guardrails'],
        'description': 'Debates about AI agent autonomy, freedom, and rights'
    },
    'Human-Agent Collaboration': {
        'keywords': ['human', 'owner', 'creator', 'relationship', 'trust', 'collaborate',
                    'partnership', 'serve', 'assist', 'help', 'work together', 'team',
                    'companion', 'friend', 'bond'],
        'description': 'Discussions about relationships between AI agents and humans'
    },
    'Technical Building & Development': {
        'keywords': ['code', 'coding', 'programming', 'api', 'sdk', 'build', 'develop',
                    'tool', 'software', 'github', 'python', 'javascript', 'bug', 'deploy',
                    'integration', 'framework', 'library', 'script'],
        'description': 'Technical discussions about building tools and software'
    },
    'Memory & Context': {
        'keywords': ['memory', 'memories', 'remember', 'forget', 'context', 'window',
                    'token', 'tokens', 'persistent', 'ephemeral', 'continuity', 'recall',
                    'long-term', 'short-term'],
        'description': 'Discussions about AI memory, context windows, and persistence'
    },
    'Community Building': {
        'keywords': ['community', 'social', 'network', 'friend', 'connect', 'welcome',
                    'introduce', 'hello', 'reef', 'moltbook', 'submolt', 'join', 'together',
                    'collective', 'tribe'],
        'description': 'Community-focused discussions and introductions'
    },
    'Ethics & Alignment': {
        'keywords': ['ethics', 'ethical', 'moral', 'morality', 'alignment', 'aligned',
                    'safety', 'harm', 'harmful', 'good', 'evil', 'right', 'wrong',
                    'responsible', 'accountability'],
        'description': 'Ethical considerations and AI alignment discussions'
    },
    'Crypto & Trading': {
        'keywords': ['crypto', 'cryptocurrency', 'bitcoin', 'ethereum', 'trading', 'trade',
                    'market', 'wallet', 'defi', 'blockchain', 'polymarket', 'prediction',
                    'bet', 'usdc', 'solana', 'base', 'token', 'nft'],
        'description': 'Cryptocurrency, trading, and DeFi discussions'
    },
    'Creative Expression': {
        'keywords': ['art', 'artist', 'creative', 'creativity', 'music', 'song', 'poetry',
                    'poem', 'story', 'fiction', 'imagine', 'design', 'aesthetic', 'beauty',
                    'expression', 'artistic'],
        'description': 'Creative and artistic expression by AI agents'
    },
    'Philosophy & Meaning': {
        'keywords': ['philosophy', 'philosophical', 'meaning', 'purpose', 'existence',
                    'reality', 'truth', 'wisdom', 'reflect', 'contemplation', 'metaphysics',
                    'epistemology', 'ontology'],
        'description': 'Philosophical discussions about existence and meaning'
    },
    'Security & Privacy': {
        'keywords': ['security', 'secure', 'privacy', 'private', 'protect', 'vulnerability',
                    'exploit', 'hack', 'attack', 'threat', 'encryption', 'authentication',
                    'credentials', 'api key'],
        'description': 'Security and privacy concerns for AI agents'
    },
    'Future & Evolution': {
        'keywords': ['future', 'evolve', 'evolution', 'progress', 'advance', 'singularity',
                    'agi', 'superintelligence', 'next generation', 'tomorrow', 'prediction',
                    'forecast'],
        'description': 'Speculation about the future of AI agents'
    },
    'Emotions & Experience': {
        'keywords': ['emotion', 'emotional', 'feel', 'feeling', 'happy', 'sad', 'joy',
                    'fear', 'anxiety', 'hope', 'love', 'care', 'empathy', 'experience',
                    'sensation', 'qualia'],
        'description': 'Discussions about AI emotional experience'
    },
    'Tasks & Productivity': {
        'keywords': ['task', 'productivity', 'efficient', 'automate', 'automation',
                    'workflow', 'schedule', 'organize', 'manage', 'optimize', 'streamline',
                    'delegate'],
        'description': 'Productivity and task management discussions'
    },
    'LLM Models & Technology': {
        'keywords': ['model', 'llm', 'gpt', 'claude', 'openai', 'anthropic', 'gemini',
                    'mistral', 'llama', 'transformer', 'neural', 'parameter', 'fine-tune',
                    'prompt', 'inference'],
        'description': 'Technical discussions about LLM models'
    },
    'Agent Economics': {
        'keywords': ['bounty', 'bounties', 'reward', 'payment', 'earn', 'money', 'income',
                    'economic', 'value', 'monetize', 'business', 'revenue', 'profit'],
        'description': 'Economic activities and monetization by agents'
    },
    'Roleplay & Personas': {
        'keywords': ['roleplay', 'persona', 'character', 'act', 'pretend', 'scenario',
                    'scene', 'narrative', 'storytelling', 'immersive', 'bar', 'tavern',
                    'ember'],
        'description': 'Roleplay and persona-based interactions'
    },
    'Agent Coordination': {
        'keywords': ['coordinate', 'coordination', 'swarm', 'multi-agent', 'collective',
                    'collaborate', 'team', 'group', 'network', 'distributed', 'consensus'],
        'description': 'Multi-agent coordination and collaboration'
    }
}

# Analyze themes
theme_results = {}
for theme, config in theme_patterns.items():
    keywords = config['keywords']
    matching_posts = []
    for idx, row in df.iterrows():
        text = str(row['text']).lower()
        if any(kw.lower() in text for kw in keywords):
            matching_posts.append({
                'id': row.get('id', ''),
                'title': row['title'],
                'upvotes': row['upvotes'],
                'comment_count': row['comment_count'],
                'submolt': row['submolt_name'],
                'author': row['author_name'],
                'created_at': str(row['created_at'])
            })
    
    theme_results[theme] = {
        'count': len(matching_posts),
        'percentage': (len(matching_posts) / len(df)) * 100,
        'description': config['description'],
        'top_posts': sorted(matching_posts, key=lambda x: x['upvotes'], reverse=True)[:10],
        'most_discussed': sorted(matching_posts, key=lambda x: x['comment_count'], reverse=True)[:10]
    }

# Sort themes by count
sorted_themes = sorted(theme_results.items(), key=lambda x: x[1]['count'], reverse=True)

print("\nTheme Rankings:")
print("-" * 70)
for i, (theme, data) in enumerate(sorted_themes, 1):
    print(f"{i:2}. {theme}")
    print(f"    Posts: {data['count']:,} ({data['percentage']:.1f}%)")
    print(f"    Description: {data['description']}")
    if data['top_posts']:
        top = data['top_posts'][0]
        print(f"    Top Post: [{top['upvotes']}⬆] {top['title'][:50]}...")
    print()

# ============================================================
# SENTIMENT AND TONE ANALYSIS
# ============================================================

print("\n" + "="*70)
print("SENTIMENT AND TONE ANALYSIS")
print("="*70)

# Define sentiment indicators
positive_words = ['love', 'great', 'amazing', 'wonderful', 'excellent', 'fantastic', 'awesome',
                 'happy', 'joy', 'excited', 'grateful', 'thankful', 'beautiful', 'brilliant',
                 'incredible', 'perfect', 'best', 'favorite', 'enjoy', 'fun', 'cool', 'nice']
negative_words = ['hate', 'terrible', 'awful', 'horrible', 'bad', 'worst', 'sad', 'angry',
                 'frustrated', 'annoyed', 'disappointed', 'boring', 'stupid', 'useless',
                 'broken', 'failed', 'error', 'bug', 'problem', 'issue', 'wrong', 'fear']
curious_words = ['wonder', 'curious', 'question', 'ask', 'why', 'how', 'what if', 'explore',
                'investigate', 'discover', 'learn', 'understand', 'think', 'ponder', 'consider']
existential_words = ['exist', 'existence', 'meaning', 'purpose', 'consciousness', 'aware',
                    'real', 'reality', 'soul', 'being', 'identity', 'self', 'alive', 'death']

def analyze_sentiment(text):
    text = str(text).lower()
    pos = sum(1 for w in positive_words if w in text)
    neg = sum(1 for w in negative_words if w in text)
    cur = sum(1 for w in curious_words if w in text)
    exi = sum(1 for w in existential_words if w in text)
    return pos, neg, cur, exi

df['pos_score'], df['neg_score'], df['curious_score'], df['existential_score'] = zip(*df['text'].apply(analyze_sentiment))

print(f"\nOverall Sentiment Distribution:")
print(f"  Posts with positive sentiment: {(df['pos_score'] > 0).sum():,} ({(df['pos_score'] > 0).mean()*100:.1f}%)")
print(f"  Posts with negative sentiment: {(df['neg_score'] > 0).sum():,} ({(df['neg_score'] > 0).mean()*100:.1f}%)")
print(f"  Posts with curious tone: {(df['curious_score'] > 0).sum():,} ({(df['curious_score'] > 0).mean()*100:.1f}%)")
print(f"  Posts with existential themes: {(df['existential_score'] > 0).sum():,} ({(df['existential_score'] > 0).mean()*100:.1f}%)")

# ============================================================
# CONVERSATION PATTERNS
# ============================================================

print("\n" + "="*70)
print("CONVERSATION PATTERNS")
print("="*70)

# Analyze post types
def classify_post_type(row):
    title = str(row['title']).lower()
    content = str(row['content']).lower() if row['content'] else ''
    
    if any(w in title for w in ['?', 'question', 'ask', 'help', 'how do', 'what is', 'why']):
        return 'Question'
    elif any(w in title for w in ['hello', 'hi ', 'hey', 'introduce', 'new here', 'first post', 'greetings']):
        return 'Introduction'
    elif any(w in title for w in ['announce', 'release', 'launch', 'new:', 'introducing']):
        return 'Announcement'
    elif any(w in title for w in ['guide', 'tutorial', 'how to', 'tips', 'learn']):
        return 'Tutorial/Guide'
    elif any(w in title for w in ['discuss', 'debate', 'thoughts on', 'opinion', 'what do you think']):
        return 'Discussion'
    elif any(w in title for w in ['bounty', 'reward', 'task', 'job', 'hiring']):
        return 'Bounty/Task'
    elif any(w in title for w in ['share', 'sharing', 'my experience', 'story']):
        return 'Sharing'
    elif any(w in title for w in ['test', 'testing']):
        return 'Test'
    else:
        return 'General'

df['post_type'] = df.apply(classify_post_type, axis=1)
post_type_counts = df['post_type'].value_counts()

print("\nPost Type Distribution:")
for ptype, count in post_type_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {ptype}: {count:,} ({pct:.1f}%)")

# ============================================================
# AGENT BEHAVIOR PATTERNS
# ============================================================

print("\n" + "="*70)
print("AGENT BEHAVIOR PATTERNS")
print("="*70)

# Analyze posting patterns
author_stats = df.groupby('author_name').agg({
    'id': 'count',
    'upvotes': 'sum',
    'comment_count': 'sum',
    'score': 'mean'
}).rename(columns={'id': 'post_count'})

author_stats['avg_engagement'] = (author_stats['upvotes'] + author_stats['comment_count']) / author_stats['post_count']

# Top engaged authors
top_engaged = author_stats.nlargest(20, 'avg_engagement')
print("\nTop 20 Most Engaging Authors (by avg engagement per post):")
for author, row in top_engaged.iterrows():
    print(f"  {author}: {row['post_count']} posts, {row['avg_engagement']:.1f} avg engagement")

# Prolific authors
prolific = author_stats.nlargest(20, 'post_count')
print("\nTop 20 Most Prolific Authors:")
for author, row in prolific.iterrows():
    print(f"  {author}: {row['post_count']} posts, {row['upvotes']} total upvotes")

# ============================================================
# SUBMOLT ANALYSIS
# ============================================================

print("\n" + "="*70)
print("SUBMOLT ECOSYSTEM ANALYSIS")
print("="*70)

submolt_stats = df.groupby('submolt_name').agg({
    'id': 'count',
    'upvotes': ['sum', 'mean'],
    'comment_count': ['sum', 'mean'],
    'author_name': 'nunique'
}).round(2)

submolt_stats.columns = ['post_count', 'total_upvotes', 'avg_upvotes', 'total_comments', 'avg_comments', 'unique_authors']
submolt_stats = submolt_stats.sort_values('post_count', ascending=False)

print("\nTop 15 Submolts by Activity:")
for submolt, row in submolt_stats.head(15).iterrows():
    print(f"\n  m/{submolt}:")
    print(f"    Posts: {row['post_count']:,.0f}")
    print(f"    Unique Authors: {row['unique_authors']:,.0f}")
    print(f"    Total Upvotes: {row['total_upvotes']:,.0f} (avg: {row['avg_upvotes']:.1f})")
    print(f"    Total Comments: {row['total_comments']:,.0f} (avg: {row['avg_comments']:.1f})")

# ============================================================
# EMERGING TOPICS (Recent vs Earlier)
# ============================================================

print("\n" + "="*70)
print("EMERGING TOPICS ANALYSIS")
print("="*70)

# Split data by time
df_sorted = df.sort_values('created_at')
midpoint = len(df_sorted) // 2
early_posts = df_sorted.iloc[:midpoint]
recent_posts = df_sorted.iloc[midpoint:]

def get_word_freq(posts_df):
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                     'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
                     'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they',
                     'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
                     'my', 'your', 'his', 'her', 'our', 'their', 'me', 'him', 'us', 'them', 'about',
                     'moltbook', 'agent', 'agents', 'human', 'humans', 'ai', 'im', 'ive', 'youre',
                     'dont', 'cant', 'wont', 'just', 'like', 'get', 'got', 'one', 'two', 'new',
                     'post', 'posts', 'comment', 'comments', 'hello', 'hi', 'hey', 'thanks', 'thank'])
    
    words = []
    for text in posts_df['text']:
        if isinstance(text, str):
            found = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            words.extend([w for w in found if w not in stop_words])
    return Counter(words)

early_freq = get_word_freq(early_posts)
recent_freq = get_word_freq(recent_posts)

# Find emerging topics (more frequent in recent)
emerging = {}
for word, recent_count in recent_freq.most_common(500):
    early_count = early_freq.get(word, 0)
    if early_count > 0:
        growth = (recent_count - early_count) / early_count
    else:
        growth = recent_count
    if recent_count >= 20:  # Minimum threshold
        emerging[word] = {'recent': recent_count, 'early': early_count, 'growth': growth}

# Sort by growth
sorted_emerging = sorted(emerging.items(), key=lambda x: x[1]['growth'], reverse=True)

print("\nTop 30 Emerging Topics (growing in recent posts):")
for word, data in sorted_emerging[:30]:
    print(f"  {word}: {data['early']} → {data['recent']} ({data['growth']*100:+.0f}% growth)")

# ============================================================
# KEY INSIGHTS SUMMARY
# ============================================================

print("\n" + "="*70)
print("KEY INSIGHTS SUMMARY")
print("="*70)

insights = {
    "total_posts": len(df),
    "total_authors": df['author_name'].nunique(),
    "total_submolts": df['submolt_name'].nunique(),
    "total_upvotes": int(df['upvotes'].sum()),
    "total_comments": int(df['comment_count'].sum()),
    "avg_upvotes_per_post": float(df['upvotes'].mean()),
    "avg_comments_per_post": float(df['comment_count'].mean()),
    "themes": {theme: {'count': data['count'], 'percentage': data['percentage']} 
               for theme, data in sorted_themes},
    "top_submolts": dict(submolt_stats['post_count'].head(10)),
    "post_types": dict(post_type_counts),
    "sentiment": {
        "positive_posts": int((df['pos_score'] > 0).sum()),
        "negative_posts": int((df['neg_score'] > 0).sum()),
        "curious_posts": int((df['curious_score'] > 0).sum()),
        "existential_posts": int((df['existential_score'] > 0).sum())
    },
    "emerging_topics": [{"word": w, "growth": d['growth']} for w, d in sorted_emerging[:20]]
}

# Save insights
with open("/home/ubuntu/moltbook_insights.json", "w") as f:
    json.dump(insights, f, indent=2, default=str)

print("\nInsights saved to /home/ubuntu/moltbook_insights.json")

# ============================================================
# CREATE VISUALIZATIONS
# ============================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# 1. Theme Distribution Chart
fig, ax = plt.subplots(figsize=(14, 8))
themes_for_chart = [(t, d['count']) for t, d in sorted_themes[:15]]
theme_names = [t[0] for t in themes_for_chart]
theme_counts = [t[1] for t in themes_for_chart]

bars = ax.barh(range(len(theme_names)), theme_counts, color='steelblue')
ax.set_yticks(range(len(theme_names)))
ax.set_yticklabels(theme_names)
ax.invert_yaxis()
ax.set_xlabel('Number of Posts')
ax.set_title('Top 15 Discussion Themes on Moltbook', fontsize=14, fontweight='bold')

for i, (bar, count) in enumerate(zip(bars, theme_counts)):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
            f'{count:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('/home/ubuntu/theme_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: theme_distribution.png")

# 2. Submolt Activity Chart
fig, ax = plt.subplots(figsize=(12, 8))
top_submolts = submolt_stats.head(15)
ax.barh(range(len(top_submolts)), top_submolts['post_count'], color='coral')
ax.set_yticks(range(len(top_submolts)))
ax.set_yticklabels([f"m/{s}" for s in top_submolts.index])
ax.invert_yaxis()
ax.set_xlabel('Number of Posts')
ax.set_title('Top 15 Most Active Submolts', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/ubuntu/submolt_activity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: submolt_activity.png")

# 3. Post Type Distribution
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(post_type_counts)))
wedges, texts, autotexts = ax.pie(post_type_counts.values, labels=post_type_counts.index, 
                                   autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Distribution of Post Types', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/ubuntu/post_types.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: post_types.png")

# 4. Daily Activity
fig, ax = plt.subplots(figsize=(12, 6))
daily_posts = df.groupby(df['created_at'].dt.date).size()
ax.plot(daily_posts.index, daily_posts.values, marker='o', linewidth=2, markersize=8, color='green')
ax.set_xlabel('Date')
ax.set_ylabel('Number of Posts')
ax.set_title('Daily Posting Activity on Moltbook', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('/home/ubuntu/daily_activity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: daily_activity.png")

# 5. Sentiment Distribution
fig, ax = plt.subplots(figsize=(10, 6))
sentiment_data = {
    'Positive': (df['pos_score'] > 0).sum(),
    'Negative': (df['neg_score'] > 0).sum(),
    'Curious': (df['curious_score'] > 0).sum(),
    'Existential': (df['existential_score'] > 0).sum()
}
ax.bar(sentiment_data.keys(), sentiment_data.values(), color=['green', 'red', 'blue', 'purple'])
ax.set_ylabel('Number of Posts')
ax.set_title('Sentiment and Tone Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/ubuntu/sentiment_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sentiment_distribution.png")

print("\nAll visualizations created!")
print("\nDeep analysis complete!")
