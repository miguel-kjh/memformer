import torch
from memformer.memformer import Memformer
from memformer.optimization.mrbp import memory_replay_backprop

def encode_only():
    model = Memformer(
    dim = 512,
    enc_num_tokens = 256,
    enc_heads = 8,
    enc_depth = 2,
    enc_max_seq_len = 1024,
    num_memory_slots = 128,
    num_mem_updates = 2,
    encoder_only = True       # only use encoder, in which output is encoded output
    )

    src1 = torch.randint(0, 256, (1, 1024))
    src2 = torch.randint(0, 256, (1, 1024))

    enc1, mems1 = model(src1) # (1, 1024, 512), (1, 128, 512)
    enc2, mems2 = model(src2, mems = mems1)

    # initial memory
    print('mems1.shape:', mems1)

    # print the results
    print('src1.shape:', src1.shape)
    print('enc1.shape:', enc1.shape)

    print('src2.shape:', src2.shape)
    print('enc2.shape:', enc2.shape)

    print('mems1.shape:', mems1.shape)
    print('mems2.shape:', mems2.shape)
    print("Encoder only done.")

def encode_decode_mrbp():
    model = Memformer(
        dim = 512,
        num_memory_slots = 128,
        enc_num_tokens = 256,
        enc_depth = 2,
        enc_max_seq_len = 1024,
        dec_num_tokens = 256,
        dec_depth = 2,
        dec_max_seq_len = 1024
    ).cuda()

    seq = torch.randint(0, 256, (1, 8192)).cuda()
    print("seq.shape:", seq.shape)
    seq_mask = torch.ones_like(seq).bool().cuda()

    tgt = torch.randint(0, 256, (1, 512)).cuda()
    print("tgt.shape:", tgt.shape)
    tgt_mask = torch.ones_like(tgt).bool().cuda()

    # will automatically split the source sequence to 8 segments
    for _ in range(8):
        loss = memory_replay_backprop(
            model,
            src = seq,
            tgt = tgt,
            src_mask = seq_mask,
            tgt_mask = tgt_mask
        )
        print(loss.item())
    print("Encoder decoder done + MRBp.")

def encode_decode():
    model = Memformer(
        dim = 512,
        enc_num_tokens = 256,
        enc_depth = 2,
        enc_heads = 8,
        enc_max_seq_len = 1024,
        dec_num_tokens = 256,
        dec_depth = 2,
        dec_heads = 8,
        dec_max_seq_len = 1024,
        num_memory_slots = 128
    )

    src_seg_1 = torch.randint(0, 256, (1, 1024))
    src_seg_2 = torch.randint(0, 256, (1, 1024))
    src_seg_3 = torch.randint(0, 256, (1, 1024))

    tgt = torch.randint(0, 256, (1, 1024))

    enc_out1, mems1,    _ = model(src_seg_1) # (1, 1024, 512), (1, 128, 512), _
    enc_out2, mems2,    _ = model(src_seg_2, mems = mems1)
    enc_out3, mems3, loss = model(src_seg_3, tgt, mems = mems2)

    loss.backward()
    print("Encoder decoder done .")

def main():
    #encode_only()
    #encode_decode()
    encode_decode_mrbp()

if __name__ == '__main__':
    main()

