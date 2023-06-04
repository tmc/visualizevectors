import pandas as pd
import json
from datetime import datetime, timezone


def is_json(myjson):
  try:
    json.loads(myjson)
  except ValueError:
    return False
  return True

chatsembeddingscsv = pd.read_csv('static/chats-embeddings-ada-002.csv')
chats = pd.read_csv('static/chats.csv')
chatText = chats['chat_text'].values
chatsembeddingscsv = chatsembeddingscsv.dropna()
chatsembeddingscsv = chatsembeddingscsv.reset_index(drop=True)
chatsembeddingscsv = chatsembeddingscsv.drop(columns=['Unnamed: 0'])
thread_id = chatsembeddingscsv['thread_id'].values
channel_names = chats['channel_name'].values
# save all vectors under embeddings into a text file as a json
embeddings = chatsembeddingscsv['embedding'].values

def load_ndjson(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data
raw_channel_data = load_ndjson('static/channels.json')

channel_name_to_ids = {}
for cd in raw_channel_data:
  if cd['name'] not in channel_name_to_ids:
    channel_name_to_ids[cd['name']] = cd['id']


def convert_large_num_to_float_str(large_num_str):
    # Convert to integer first
    large_num = int(large_num_str)
    # Then convert to float with the division
    float_num = large_num / 1e6
    # Format the floating number to string with 6 decimal places
    float_num_str = "{:.6f}".format(float_num)
    return float_num_str

def convert_datetime_to_microtimestamp(dt_str):
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f %Z")
    except ValueError:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S %Z")

    timestamp_in_seconds = dt.replace(tzinfo=timezone.utc).timestamp()
    #return convert_large_num_to_float_str(int(timestamp_in_seconds * 1000000))
    return str(int(timestamp_in_seconds * 1000000))

thread_ids = [convert_datetime_to_microtimestamp(tid) for tid in chatsembeddingscsv['thread_id'].values]


print("thread ids:", len(thread_ids))
print("thread ids:", len(thread_id))

# print first 10 thread_ids:
print(thread_ids[:10])

result= {}

allow_list = set([
  #'general',
  'computer-vision',
  #'atlanta',
  #'founder-dating',
  'responsible-ai',
  #'content-archive',
  #'oss-evidently',
  'data-engineering',
  #'amsterdam-meetup-organization',
  #'korea',
  'vendor-info',
  #'amsterdam',
  #'nordics-public',
  'security-n-privacy',
  'gcp',
  'news',
  #'munchen',
  #'switzerland',
  'aws',
  'recsys',
  #'women-of-mlops',
  #'london',
  #'nyc',
  #'code-pals',
  #'random',
  #'manchester',
  #'mlops-alerts',
  #'officehours',
  #'wg-community-wiki',
  #'events',
  #'boxkite-monitoring',
  #'tiktok-crew',
  #'washington-dc-the-capital',
  #'cairo',
  #'berlin',
  'learning-resources',
  #'oss-zenml',
  #'spanish-speaking',
  #'israel',
  #'kudos',
  #'virtual-coffee',
  #'bristol',
  #'turkey',
  'datascience',
  #'india',
  'edge-mlops',
  #'toronto',
  # 'healthcare', # TODO
  #'sydney',
  #'boston',
#'scotland',
'discussions',
#'los-angeles',
'monitoring-ml',
'reading-group',
#'clearml-combinatorml',
#'españa',
'nlp',
'engineeringlabs',
#'meta-channel',
#'portugal',
#'greece',
#'kubecon',
#'oss-clearml',
'vertex-ai',
#'oss-gitlab',
'azure',
'leadership',
#'berlin-organizers-dh',
#'oss-anovos',
#'career-advice',
#'neptune-ai',
#'bay-area',
#'fiddler',
#'suggestspeakers',
#'marketing-mlops',
#'montreal',
'production-code',
'generative-ai',
'feature-store',
'dataops',
#'ama-sessions',
#'aws-rants',
#'montreal-organizers',
#'french-speaking',
#'bad-startup-ideas',
#'berlin-conf-speaker-selection',
'building-stuff',
#'mlops_survery_data_visualisation',
#'chicago',
#'berlin-organizers',
'llmops',
#'seattle',
'machine-learning-interview',
'mlops-world',
#'cats',
#'evangelism',
#'pancake-stacks',
#'api-sdk',
#'copenhagen',
#'oss-modelstore',
#'introduce-yourself',
#'vc-dating',
'kubernetes',
'labeling',
#'colorado',
#'chassis-model-builder',
#'cloudstores',
#'middle-east',
#'fresh-jamz',
#'africa',
#'paris',
'testing-ml',
#'be-shameless',
#'community-conferences',
'mlops-questions-answered',
'open-source',
'blog',
#'swag-ideas',
'mlops-research',
'arize-ai',
#'meetup-barcelona',
'mlops-stacks',
'data-pipelines-orchestration',
#'project-managers',
#'jobs'
  ])

for i in range(len(embeddings)):
    channel_name = channel_names[i]
    if channel_name not in allow_list:
      continue
    channel_id = channel_name_to_ids[channel_name]
    tid = thread_ids[i]
    result[tid] = {
        "thread_id": tid,
        "url": f"https://mlops-community.slack.com/archives/{channel_id}/p{tid}",
        "embedding": embeddings[i],
        "chat": chatText[i]
    }
    
# print a few recroeds from results:
print(list(result.items())[:10])



with open('static/embeddings.txt', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
