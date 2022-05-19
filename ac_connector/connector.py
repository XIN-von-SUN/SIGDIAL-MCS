# This function takes in the model and character as arguments and returns the next character prediction and hidden state
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        self.dropout = nn.Dropout(0.05)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        #print('batch_size: ', x.size())
        batch_size = x.size(0)
        
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        #print('inputs: ', x.size())

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        #print(out.size())
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(out)
        out = self.fc(out)
        #print('output: ', out.size())
        #print(out.size())
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
        

def embeddings(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features


def predict(model, operation, oper2ind, ind2oper, dict_size, embeddings):
    # One-hot encoding our input to fit into the model
    
    operation = np.array([[oper2ind[oper] for oper in operation]])
    operation = embeddings(operation, dict_size, operation.shape[1], 1)
    operation = torch.from_numpy(operation)
    operation.to(device)
    
    out, hidden = model(operation)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    oper_ind = torch.max(prob, dim=0)[1].item()

    return ind2oper[oper_ind], hidden


# This function takes the desired output length and input characters as arguments, returning the produced sentence
# This function will output next out_len prediction results
def sample(model, oper2ind, ind2oper, dict_size, out_len, start):
    model.eval() # eval mode
    #start = start.lower()
    # First off, run through the starting characters
    opers = [oper for oper in start]
    size = out_len - len(opers)
    res = []
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        pred_oper, h = predict(model, opers, oper2ind, ind2oper, dict_size, embeddings)
        #if pred_oper == 'PAD_IDX':
         #   break
        opers.append(pred_oper)
        res.append(pred_oper)
    return res #', '.join(res)


def load(file_oper2ind, file_ind2oper):
    oper2ind = np.load(file_oper2ind, allow_pickle=True).item()
    ind2oper = np.load(file_ind2oper, allow_pickle=True).item()
    
    return oper2ind, ind2oper


def connector(model, oper2ind, ind2oper, dict_size, talked_pipelines, out_len):
    if len(talked_pipelines) > 0:
        pass
    else:
        talked_pipelines = ["switch_greeting"]

    # print(f'talked_pipelines is: {talked_pipelines}')

    pred_pipelines = sample(model, oper2ind, ind2oper, dict_size, out_len, talked_pipelines)
    # print(f'pred_pipelines is: {pred_pipelines}')
    
    not_available_pipelines = ['switch_memory_recall', 'switch_goal_setting', 'switch_motivator']

    for i in pred_pipelines:
        if i not in talked_pipelines and i != 'PAD_IDX' and i not in not_available_pipelines:
            next_pipeline = i
            talked_pipelines.append(next_pipeline)
            # print(f'next_pipeline is: {next_pipeline} \n')

            # here will set the slot True value for 'next_pipeline'
            utter_dict = {
                    "switch_greeting": "utter_ask_permission_topic_greeting",
                    "switch_pa": "utter_ask_permission_topic_pa",
                    "switch_step_count": "utter_ask_permission_step_count",
                    "switch_memory_recall": "utter_ask_permission_memory_recall",
                    "switch_goal_setting": "utter_ask_permission_goal_setting",
                    "switch_rating_importance": "utter_ask_permission_topic_rating_importance", 
                    "switch_rating_confidence": "utter_ask_permission_topic_rating_confidence",
                    "switch_self_efficacy": "utter_ask_permission_topic_self_efficacy",
                    "switch_motivator": "utter_ask_permission_topic_motivator",
                    }
            # utter_stop_current_topic = f"Seems we can move on to the next topic!"
            response = utter_dict[next_pipeline]
            return talked_pipelines, response, next_pipeline
            break
        elif i in talked_pipelines or i in not_available_pipelines:
            pass
        elif i == 'PAD_IDX':
            next_pipeline = None
            # print(f'next_pipeline is: {next_pipeline}')
            
            utter_no_topic_left = f"It seems we have talked a lot! I will catch you up next time! See you soon!"
            # print(utter_no_topic_left,  '\n')
            # dispatcher.utter_message(response="utter_no_topic_left")
            response = "utter_no_topic_left"
            # return [SlotSet("all_topic", all_topic), SlotSet("next_topic", None)]
            return talked_pipelines, response, next_pipeline
            break


def connector_run(current_path, talked_pipelines, out_len=6):
    # load
    file_oper2ind = current_path + '/oper_ind/oper2ind.npy'
    file_ind2oper = current_path + '/oper_ind/ind2oper.npy'

    oper2ind, ind2oper = load(file_oper2ind, file_ind2oper)

    dict_size = len(oper2ind)
    PATH = current_path + '/model/model_rnn.pt'
    model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=64, n_layers=1)   # TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    print(f'talked_pipelines 1 is: {talked_pipelines} \n')
    talked_pipelines, response, next_pipeline = connector(model, oper2ind, ind2oper, dict_size, talked_pipelines, out_len)  
    return talked_pipelines, response, next_pipeline



if __name__=="__main__":

    # # load
    # file_oper2ind = './oper_ind/oper2ind.npy'
    # file_ind2oper = './oper_ind/ind2oper.npy'
    # oper2ind, ind2oper = load(file_oper2ind, file_ind2oper)

    # dict_size = len(oper2ind)
    # PATH = './model/model_rnn.pt'
    # model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=64, n_layers=1)   # TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()

    """
    # test
    res1 = sample(model, 5, ['switch_greeting', 'switch_pa'])
    res2 = sample(model, 5, ['switch_greeting', 'switch_pa', 'switch_rating_importance'])
    res3 = sample(model, 5, ['switch_rating_importance', 'switch_rating_confidence'])
    res4 = sample(model, 5, ['switch_step_count', 'switch_goal_setting'])
    res5 = sample(model, 5, ['switch_greeting'])

    print('res1: ', res1, '\n')
    print('res2: ', res2, '\n')
    print('res3: ', res3, '\n')
    print('res4: ', res4, '\n')
    print('res5: ', res5, '\n')

    response1 = connector(['switch_greeting', 'switch_pa'])
    response2 = connector(['switch_greeting', 'switch_pa', 'switch_rating_importance'])
    response3 = connector(['switch_rating_importance', 'switch_rating_confidence'])
    response4 = connector(['switch_step_count', 'switch_goal_setting'])    
    response5 = connector([])

    print('response1: ', response1, '\n')
    print('response2: ', response2, '\n')
    print('response3: ', response3, '\n')
    print('response4: ', response4, '\n')
    print('response5: ', response5, '\n')
    """
    
    # talked_pipelines = ['switch_greeting']
    # for i in range(6):
    #     talked_pipelines, response, next_pipeline = connector(talked_pipelines, out_len=6)  
    #     # print(f'talked_pipelines is: {talked_pipelines}')
    current_path = os.getcwd()
    talked_pipelines = ['switch_greeting']
    for i in range(6):
        talked_pipelines, response, next_pipeline = connector_run(current_path, talked_pipelines, out_len=6)
        print(f'next_pipeline is: {next_pipeline}')
        print(f'talked_pipelines is: {talked_pipelines} \n')

  
