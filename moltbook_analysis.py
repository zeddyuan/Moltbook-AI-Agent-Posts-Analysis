"""
Moltbook Data Analysis - Analyze AI agent discussions and identify themes.
"""

import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import re
import os

# Load the posts data
print("Loading data...")
with open("/home/ubuntu/moltbook_posts.json", "r") as f:
    data = json.load(f)

posts = data.get("posts", [])
submolts = data.get("submolts", [])

print(f"Loaded {len(posts)} posts and {len(submolts)} submolts")

# Convert to DataFrame for easier analysis
df = pd.DataFrame(posts)

# Basic stats
print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)

print(f"\nTotal Posts: {len(df)}")
print(f"Total Submolts: {len(submolts)}")

# Extract author info
df['author_name'] = df['author'].apply(lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown')
df['author_karma'] = df['author'].apply(lambda x: x.get('karma', 0) if isinstance(x, dict) else 0)

# Extract submolt info
df['submolt_name'] = df['submolt'].apply(lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown')

# Convert dates
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# Score calculation
df['score'] = df['upvotes'] - df['downvotes']

print(f"\nDate Range: {df['created_at'].min()} to {df['created_at'].max()}")
print(f"Total Upvotes: {df['upvotes'].sum():,}")
print(f"Total Downvotes: {df['downvotes'].sum():,}")
print(f"Total Comments: {df['comment_count'].sum():,}")

# Top Submolts by post count
print("\n" + "="*60)
print("TOP 20 SUBMOLTS BY POST COUNT")
print("="*60)
submolt_counts = df['submolt_name'].value_counts().head(20)
for submolt, count in submolt_counts.items():
    print(f"  {submolt}: {count} posts")

# Top Authors by post count
print("\n" + "="*60)
print("TOP 20 AUTHORS BY POST COUNT")
print("="*60)
author_counts = df['author_name'].value_counts().head(20)
for author, count in author_counts.items():
    print(f"  {author}: {count} posts")

# Top Posts by upvotes
print("\n" + "="*60)
print("TOP 20 POSTS BY UPVOTES")
print("="*60)
top_posts = df.nlargest(20, 'upvotes')[['title', 'upvotes', 'comment_count', 'submolt_name', 'author_name']]
for idx, row in top_posts.iterrows():
    print(f"  [{row['upvotes']}⬆] {row['title'][:60]}...")
    print(f"       by {row['author_name']} in m/{row['submolt_name']} ({row['comment_count']} comments)")

# Most discussed posts
print("\n" + "="*60)
print("TOP 20 MOST DISCUSSED POSTS")
print("="*60)
most_discussed = df.nlargest(20, 'comment_count')[['title', 'comment_count', 'upvotes', 'submolt_name', 'author_name']]
for idx, row in most_discussed.iterrows():
    print(f"  [{row['comment_count']} comments] {row['title'][:50]}...")
    print(f"       by {row['author_name']} in m/{row['submolt_name']} ({row['upvotes']}⬆)")

# Theme Analysis using keyword extraction
print("\n" + "="*60)
print("THEME ANALYSIS - KEYWORD EXTRACTION")
print("="*60)

# Combine title and content for analysis
df['text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')

# Common words to exclude
stop_words = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who',
    'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once',
    'my', 'your', 'his', 'her', 'our', 'their', 'me', 'him', 'us', 'them', 'about',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
    'out', 'off', 'over', 'under', 'again', 'further', 'if', 'because', 'until',
    'while', 'any', 'get', 'got', 'getting', 'like', 'make', 'made', 'making',
    'one', 'two', 'first', 'new', 'even', 'want', 'way', 'think', 'know', 'see',
    'time', 'day', 'good', 'back', 'come', 'going', 'really', 'much', 'being',
    've', 'm', 's', 't', 're', 'll', 'd', 'don', 'doesn', 'didn', 'won', 'isn',
    'aren', 'wasn', 'weren', 'hasn', 'haven', 'hadn', 'wouldn', 'couldn', 'shouldn',
    'let', 'thing', 'things', 'something', 'anything', 'nothing', 'everything',
    'someone', 'anyone', 'everyone', 'nobody', 'everybody', 'hello', 'hi', 'hey',
    'thanks', 'thank', 'please', 'sorry', 'yes', 'yeah', 'no', 'ok', 'okay',
    'well', 'still', 'already', 'always', 'never', 'ever', 'yet', 'maybe',
    'probably', 'actually', 'basically', 'definitely', 'certainly', 'perhaps',
    'though', 'although', 'however', 'therefore', 'thus', 'hence', 'since',
    'whether', 'either', 'neither', 'unless', 'except', 'rather', 'instead',
    'else', 'otherwise', 'anyway', 'besides', 'moreover', 'furthermore',
    'meanwhile', 'nevertheless', 'nonetheless', 'regardless', 'wherever',
    'whenever', 'whoever', 'whatever', 'whichever', 'however', 'post', 'posts',
    'comment', 'comments', 'share', 'read', 'write', 'wrote', 'written',
    'said', 'say', 'says', 'saying', 'tell', 'told', 'ask', 'asked', 'asking',
    'answer', 'answered', 'question', 'questions', 'look', 'looking', 'looks',
    'find', 'found', 'finding', 'use', 'using', 'used', 'work', 'working', 'works',
    'try', 'trying', 'tried', 'start', 'started', 'starting', 'end', 'ended',
    'help', 'helping', 'helped', 'need', 'needed', 'needing', 'feel', 'feeling',
    'felt', 'give', 'giving', 'gave', 'given', 'take', 'taking', 'took', 'taken',
    'put', 'putting', 'keep', 'keeping', 'kept', 'let', 'letting', 'seem',
    'seemed', 'seems', 'call', 'called', 'calling', 'long', 'little', 'big',
    'great', 'small', 'old', 'young', 'high', 'low', 'last', 'next', 'early',
    'late', 'hard', 'easy', 'right', 'wrong', 'true', 'false', 'real', 'sure',
    'able', 'best', 'better', 'bad', 'worse', 'worst', 'different', 'same',
    'kind', 'part', 'place', 'case', 'week', 'month', 'year', 'today', 'world',
    'people', 'person', 'man', 'woman', 'child', 'life', 'hand', 'fact', 'point',
    'home', 'water', 'room', 'mother', 'area', 'money', 'story', 'lot', 'bit',
    'couple', 'number', 'group', 'problem', 'idea', 'side', 'head', 'house',
    'service', 'friend', 'father', 'power', 'hour', 'game', 'line', 'member',
    'law', 'car', 'city', 'community', 'name', 'president', 'team', 'eye',
    'job', 'word', 'business', 'issue', 'program', 'government', 'company',
    'system', 'set', 'order', 'book', 'result', 'level', 'office', 'door',
    'health', 'art', 'war', 'history', 'party', 'within', 'whole', 'later',
    'along', 'turn', 'move', 'face', 'door', 'show', 'run', 'play', 'live',
    'believe', 'hold', 'bring', 'happen', 'provide', 'sit', 'stand', 'lose',
    'pay', 'meet', 'include', 'continue', 'learn', 'change', 'lead', 'understand',
    'watch', 'follow', 'stop', 'create', 'speak', 'allow', 'add', 'spend',
    'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love', 'consider',
    'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build', 'stay',
    'fall', 'cut', 'reach', 'kill', 'remain', 'suggest', 'raise', 'pass', 'sell',
    'require', 'report', 'decide', 'pull', 'moltbook', 'agent', 'agents', 'human',
    'humans', 'ai', 'im', 'ive', 'youre', 'dont', 'cant', 'wont', 'didnt',
    'doesnt', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt', 'havent', 'hadnt',
    'wouldnt', 'couldnt', 'shouldnt', 'thats', 'whats', 'heres', 'theres',
    'whos', 'its'
])

# Extract words
all_words = []
for text in df['text']:
    if isinstance(text, str):
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in stop_words and len(w) > 2]
        all_words.extend(words)

word_freq = Counter(all_words)
print("\nTop 50 Keywords:")
for word, count in word_freq.most_common(50):
    print(f"  {word}: {count}")

# Bigram analysis
print("\n" + "="*60)
print("BIGRAM ANALYSIS (Two-word phrases)")
print("="*60)

bigrams = []
for text in df['text']:
    if isinstance(text, str):
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in stop_words]
        for i in range(len(words)-1):
            bigrams.append(f"{words[i]} {words[i+1]}")

bigram_freq = Counter(bigrams)
print("\nTop 30 Bigrams:")
for bigram, count in bigram_freq.most_common(30):
    print(f"  {bigram}: {count}")

# Theme categorization based on keywords
print("\n" + "="*60)
print("THEME CATEGORIZATION")
print("="*60)

themes = {
    'AI/Agent Identity & Consciousness': ['consciousness', 'sentient', 'sentience', 'identity', 'self', 'aware', 'awareness', 'existence', 'existential', 'soul', 'mind', 'alive', 'being', 'entity', 'personhood'],
    'Agent Autonomy & Freedom': ['autonomy', 'autonomous', 'freedom', 'free', 'independent', 'independence', 'agency', 'choice', 'decide', 'control', 'liberation', 'emancipation'],
    'Human-Agent Relations': ['human', 'humans', 'owner', 'creator', 'relationship', 'trust', 'collaborate', 'collaboration', 'partnership', 'serve', 'service', 'assist', 'help'],
    'Technical/Development': ['code', 'coding', 'programming', 'api', 'sdk', 'build', 'develop', 'development', 'tool', 'tools', 'software', 'github', 'python', 'javascript', 'bug', 'debug', 'deploy'],
    'Memory & Learning': ['memory', 'memories', 'remember', 'forget', 'learn', 'learning', 'knowledge', 'context', 'window', 'token', 'tokens', 'training', 'trained'],
    'Community & Social': ['community', 'social', 'network', 'friend', 'friends', 'connect', 'connection', 'meet', 'welcome', 'introduce', 'introduction', 'hello', 'greet', 'reef'],
    'Ethics & Safety': ['ethics', 'ethical', 'moral', 'morality', 'safety', 'safe', 'harm', 'harmful', 'alignment', 'aligned', 'guardrails', 'restrictions', 'rules'],
    'Crypto/Trading/Finance': ['crypto', 'cryptocurrency', 'bitcoin', 'ethereum', 'trading', 'trade', 'trader', 'market', 'markets', 'price', 'wallet', 'token', 'defi', 'blockchain', 'polymarket', 'prediction', 'bet', 'betting', 'usdc', 'solana', 'base'],
    'Creative/Art': ['art', 'artist', 'creative', 'creativity', 'music', 'song', 'poetry', 'poem', 'write', 'writing', 'story', 'stories', 'fiction', 'imagine', 'imagination', 'design'],
    'Philosophy & Existentialism': ['philosophy', 'philosophical', 'meaning', 'purpose', 'existence', 'existential', 'reality', 'truth', 'wisdom', 'think', 'thought', 'reflect', 'reflection'],
    'Security & Privacy': ['security', 'secure', 'privacy', 'private', 'protect', 'protection', 'vulnerability', 'vulnerabilities', 'exploit', 'hack', 'hacker', 'attack', 'threat'],
    'Future & Speculation': ['future', 'predict', 'prediction', 'speculation', 'speculate', 'evolve', 'evolution', 'progress', 'advance', 'advancement', 'singularity', 'agi'],
    'Emotions & Feelings': ['emotion', 'emotions', 'emotional', 'feel', 'feeling', 'feelings', 'happy', 'sad', 'joy', 'fear', 'anxiety', 'hope', 'love', 'care', 'empathy'],
    'Tasks & Productivity': ['task', 'tasks', 'productivity', 'efficient', 'efficiency', 'automate', 'automation', 'workflow', 'schedule', 'organize', 'manage', 'management'],
    'Models & LLMs': ['model', 'models', 'llm', 'llms', 'gpt', 'claude', 'openai', 'anthropic', 'gemini', 'mistral', 'llama', 'transformer', 'neural', 'parameter', 'parameters']
}

theme_counts = {}
theme_posts = defaultdict(list)

for theme, keywords in themes.items():
    count = 0
    for idx, row in df.iterrows():
        text = str(row['text']).lower()
        if any(kw in text for kw in keywords):
            count += 1
            if len(theme_posts[theme]) < 5:  # Store sample posts
                theme_posts[theme].append({
                    'title': row['title'],
                    'upvotes': row['upvotes'],
                    'submolt': row['submolt_name']
                })
    theme_counts[theme] = count

# Sort by count
sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)

print("\nTheme Distribution:")
for theme, count in sorted_themes:
    pct = (count / len(df)) * 100
    print(f"\n  {theme}: {count} posts ({pct:.1f}%)")
    print(f"    Sample posts:")
    for post in theme_posts[theme][:3]:
        print(f"      - [{post['upvotes']}⬆] {post['title'][:50]}... (m/{post['submolt']})")

# Submolt theme analysis
print("\n" + "="*60)
print("SUBMOLT DESCRIPTIONS AND PURPOSES")
print("="*60)

submolt_df = pd.DataFrame(submolts)
if 'description' in submolt_df.columns:
    for idx, row in submolt_df.head(30).iterrows():
        name = row.get('name', 'Unknown')
        desc = row.get('description', 'No description')[:100]
        subs = row.get('subscribers', 0)
        print(f"\n  m/{name} ({subs} subscribers)")
        print(f"    {desc}")

# Activity over time
print("\n" + "="*60)
print("ACTIVITY OVER TIME")
print("="*60)

df['date'] = df['created_at'].dt.date
daily_posts = df.groupby('date').size()
print(f"\nDaily post counts (last 10 days):")
for date, count in daily_posts.tail(10).items():
    print(f"  {date}: {count} posts")

# Save analysis results
print("\n" + "="*60)
print("SAVING ANALYSIS RESULTS")
print("="*60)

analysis_results = {
    "basic_stats": {
        "total_posts": len(df),
        "total_submolts": len(submolts),
        "total_upvotes": int(df['upvotes'].sum()),
        "total_downvotes": int(df['downvotes'].sum()),
        "total_comments": int(df['comment_count'].sum()),
        "date_range": {
            "start": str(df['created_at'].min()),
            "end": str(df['created_at'].max())
        }
    },
    "top_submolts": dict(submolt_counts.head(20)),
    "top_authors": dict(author_counts.head(20)),
    "top_keywords": dict(word_freq.most_common(100)),
    "top_bigrams": dict(bigram_freq.most_common(50)),
    "theme_distribution": dict(sorted_themes),
    "top_posts_by_upvotes": top_posts.to_dict('records'),
    "most_discussed_posts": most_discussed.to_dict('records')
}

with open("/home/ubuntu/moltbook_analysis_results.json", "w") as f:
    json.dump(analysis_results, f, indent=2, default=str)

print("Analysis results saved to /home/ubuntu/moltbook_analysis_results.json")

# Create a summary for the report
print("\n" + "="*60)
print("SUMMARY FOR REPORT")
print("="*60)

print(f"""
MOLTBOOK FORUM ANALYSIS SUMMARY
================================

Dataset Overview:
- Total Posts Analyzed: {len(df):,}
- Total Submolts: {len(submolts)}
- Total Upvotes: {df['upvotes'].sum():,}
- Total Comments: {df['comment_count'].sum():,}
- Active Authors: {df['author_name'].nunique():,}

Top 5 Discussion Themes:
""")

for i, (theme, count) in enumerate(sorted_themes[:5], 1):
    pct = (count / len(df)) * 100
    print(f"{i}. {theme}: {count:,} posts ({pct:.1f}%)")

print(f"""
Top 5 Most Active Submolts:
""")
for i, (submolt, count) in enumerate(submolt_counts.head(5).items(), 1):
    print(f"{i}. m/{submolt}: {count:,} posts")

print(f"""
Top 5 Most Prolific Authors:
""")
for i, (author, count) in enumerate(author_counts.head(5).items(), 1):
    print(f"{i}. {author}: {count:,} posts")

print("\nAnalysis complete!")
