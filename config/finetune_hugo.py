# train a miniature character-level hugo model
# good for debugging and playing on macbooks and such

out_dir = 'out-hugo'
eval_interval = 50  # keep frequent because we'll overfit
eval_iters = 20
log_interval = 100  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = 'hugo'
wandb_run_name = 'mini-gpt'

dataset = 'hugo'
# for fine tuning
gradient_accumulation_steps = 32
# reduce batch size
batch_size = 8
block_size = 1024  # context of up to 1024 previous characters

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.01

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 50000
lr_decay_iters = 50000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
decay_lr = False
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
init_from = 'resume'  # 'scratch' or 'resume' or 'gpt2*'
