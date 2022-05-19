import yaml, os
import pandas as pd  

####################################################################################################################################################################

# function used to transform responses.yml into doamin.yml
def merge_responses(stories_path_list, domain_path):
    all_responses = []
    for story_path in stories_path_list:
        story_file = open(story_path, 'r')
        story =  story_file.readlines()
        try:
            idx = story.index('responses:\n')
            responses = story[idx+1:]
            responses.append('\n')
            story_file.close()
            for i in responses:
                if i.strip().startswith('#'):
                    responses.remove(i)
            all_responses.append(responses)
            story_file.close()
        except ValueError:
            pass
    all_responses.insert(0, 'responses:\n')

    domain_file = open(domain_path, 'r')
    domain = domain_file.readlines()    
    try:
        responses_idx, actions_idx = domain.index('responses:\n'), domain.index('actions:\n')
        # print(responses_idx, actions_idx)
        domain_pre = domain[:responses_idx]
        domain_post = domain[actions_idx:]
        domain_post.insert(0, '\n\n')
        domain = domain_pre + all_responses + domain_post
    except ValueError:
        pass
    
    save_file = open(domain_path, 'w+')
    for line in domain:
        save_file.writelines(line)
    save_file.close()

####################################################################################################################################################################

# function used to merge separate stories.ymls into final stories.yml
def merge_stories(stories_path_list, write_path):
    all_stories = []
    for story_path in stories_path_list:
        story_file = open(story_path, 'r')
        story =  story_file.readlines()
        for line in story[1:]:
            if line != 'responses:\n':
                all_stories.append(line)
            else:
                break
        story_file.close()
    
    all_stories.insert(0, 'version: "2.0"\n\nstories:\n')
    save_file = open(write_path, 'w+')
    for line in all_stories:
        save_file.writelines(line)
    save_file.close()

####################################################################################################################################################################

# function used to merge separate nlu.ymls into final nlu.yml
def load_yaml(load_path):
    responses_file = open(load_path, 'r')
    responses = responses_file.read()
    responses = yaml.safe_load(responses)
    return responses

def merge_nlu(nlu_path_list, write_path):
    nlu = load_yaml(nlu_path_list[0]) 
    df = pd.DataFrame(nlu['nlu'])
    for nlu_path in nlu_path_list[1:]:
        nlu = load_yaml(nlu_path) 
        df_raw = pd.DataFrame(nlu['nlu'])
        df = pd.merge(df, df_raw, on=['intent'])
        df['examples'] = df[['examples_x', 'examples_y']].apply(lambda x: x['examples_x']+x['examples_y'], axis=1)
        df = df.drop(['examples_x', 'examples_y'], axis=1)

    all_nlu = []
    all_nlu.insert(0, 'version: "2.0"\nnlu:\n')
    for i in range(len(df)):
        all_nlu.append(f'- intent: {df.loc[i, "intent"]}\n')
        all_nlu.append(f'  examples: |\n{df.loc[i, "examples"].replace("- ", "    - ")}')

    save_file = open(write_path, 'w+')
    for line in all_nlu:
        save_file.writelines(line)
    save_file.close()

####################################################################################################################################################################

# # function used to transfer /data/domain_editor.yml into domain.yml
# def restore_domain(domian_load_path, write_path):

#     domain_file = open(domian_load_path, 'r')
#     domain =  domain_file.readlines()

#     save_file = open(write_path, 'w+')
#     for line in domain:
#         save_file.writelines(line)
#     save_file.close()

####################################################################################################################################################################

if __name__ == '__main__':
    current_path = os.getcwd()
    
    # merge stories into one stories.yml
    stories_path_list = [
                        current_path + '/data/stories_responses/pipe_general.yml',
                        current_path + '/data/stories_responses/pipe_connector.yml',
                        current_path + '/data/stories_responses/pipe_goal_setting.yml',
                        current_path + '/data/stories_responses/pipe_greeting.yml',
                        current_path + '/data/stories_responses/pipe_pa.yml',
                        current_path + '/data/stories_responses/pipe_rating_importance.yml',
                        current_path + '/data/stories_responses/pipe_rating_confidence.yml',
                        current_path + '/data/stories_responses/pipe_self_efficacy.yml',
                        current_path + '/data/stories_responses/pipe_step_count.yml',
                        ]
    write_stories_path = current_path + '/data/stories.yml'
    merge_stories(stories_path_list, write_stories_path)


    # merge nlu into one nlu.yml
    nlu_path_list = [current_path + '/data/nlu_multilingual/nlu_en.yml']
    write_nlu_path = current_path + '/data/nlu.yml'
    merge_nlu(nlu_path_list, write_nlu_path)


    # merge responses into domain
    stories_path_list = [
                        current_path + '/data/stories_responses/pipe_general.yml',
                        current_path + '/data/stories_responses/pipe_connector.yml',
                        current_path + '/data/stories_responses/pipe_goal_setting.yml',
                        current_path + '/data/stories_responses/pipe_greeting.yml',
                        current_path + '/data/stories_responses/pipe_pa.yml',
                        current_path + '/data/stories_responses/pipe_rating_importance.yml',
                        current_path + '/data/stories_responses/pipe_rating_confidence.yml',
                        current_path + '/data/stories_responses/pipe_self_efficacy.yml',
                        current_path + '/data/stories_responses/pipe_step_count.yml',
                        ]    
    domain_write_path  = current_path + '/domain.yml'  # '/data/domain_editor.yml'
    merge_responses(stories_path_list, domain_write_path)


    # # transfor domain.yml
    # domain_editor_path  = current_path + '/data/domain_editor.yml'
    # domain_write_path  = current_path + '/domain.yml'
    # restore_domain(domain_editor_path, domain_write_path)
