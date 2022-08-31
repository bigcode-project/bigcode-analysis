import torch

class MultiHead:
    '''
    bs = batch size
    sl = sequence length
    nh = number of heads
    hs = head size
    nm = number of embeddings = nh * hs
    '''

    @classmethod
    def allocate_data(cls, bs, sl, nh, hs, print_shapes):
        nm = nh * hs 
        hidden_state = torch.randn(bs, sl, nm, device=torch.device('cuda'))
        c_attn_w = torch.randn(nm, 3*nm, device=torch.device('cuda'))
        i0 = None
        i1 = None
        i2 = None
        if print_shapes:
            print('hidden_state', hidden_state.shape)
            print('c_attn_w', c_attn_w.shape)
        return hidden_state, c_attn_w, i0, i1, i2

    @classmethod
    def get_qkv(cls, hidden_state, c_attn_w, i0, bs, sl, nh, hs, print_shapes):
        return torch.matmul(
            hidden_state.view(bs * sl, nh * hs),
            c_attn_w
        ).view(bs, sl, -1)

    @classmethod
    def split_qkv(cls, qkv, bs, sl, nh, hs, print_shapes):
        q, k, v = qkv.split(nh*hs, dim=2)
        
        if print_shapes:
            print('q', q.shape)
            print('k', k.shape)
            print('v', v.shape)

        q = q.view(bs, sl, nh, hs).permute(0, 2, 1, 3)
        k = k.view(bs, sl, nh, hs).permute(0, 2, 3, 1)
        v = v.view(bs, sl, nh, hs).permute(0, 2, 1, 3)
        
        if print_shapes:
            print('q', q.shape)
            print('k', k.shape)
            print('v', v.shape)
        
        return q, k, v

    @classmethod
    def get_attention_weights(cls, q, k, i1, bs, sl, nh, hs, print_shapes):
        attention_weights = torch.matmul(q, k)
        return attention_weights
    
    @classmethod
    def get_attention_output(cls, attention_weights, v, i2, bs, sl, nh, hs, print_shapes):
        attn_output = torch.matmul(attention_weights, v)
        if print_shapes:
            print('attn_output', attn_output.shape)
        #attn_output = attn_output.view(
        #    bs, nh, sl, hs).permute(0, 2, 1, 3)
        return attn_output
    
    
class MultiQuery:
    '''
    bs = batch size
    sl = sequence length
    nh = number of heads
    hs = head size
    nm = number of embeddings = nh * hs
    '''

    @classmethod
    def allocate_data(cls, bs, sl, nh, hs, print_shapes):
        nm = nh * hs 
        hidden_state = torch.randn(sl, bs, nm, device=torch.device('cuda'))
        c_attn_w = torch.randn((nh + 2) * hs, nm, device=torch.device('cuda'))
        i0 = torch.zeros((nh + 2) * hs, sl * bs, device=torch.device('cuda'))
        i1 = torch.zeros(bs, sl * nh, sl, device=torch.device('cuda'))
        i2 = torch.zeros(bs, sl * nh, hs, device=torch.device('cuda'))
        if print_shapes:
            print('hidden_state', hidden_state.shape)
            print('c_attn_w', c_attn_w.shape)
            print('i0', i0.shape)
            print('i1', i1.shape)
            print('i2', i2.shape)
        return hidden_state, c_attn_w, i0, i1, i2

    @classmethod
    def get_qkv1(cls, hidden_state, c_attn_w, i0, bs, sl, nh, hs, print_shapes):
        return torch.addmm(
            i0,
            c_attn_w,
            hidden_state.transpose(0, 1)
        )
    
    @classmethod
    def get_qkv(cls, hidden_state, c_attn_w, i0, bs, sl, nh, hs, print_shapes):
        return torch.matmul(
            c_attn_w,
            hidden_state.view(sl * bs, nh * hs).transpose(0, 1)
        )

    @classmethod
    def split_qkv(cls, qkv, bs, sl, nh, hs, print_shapes):
        q, k, v = qkv.split((nh*hs, hs, hs), dim=0)
        
        if print_shapes:
            print('q', q.shape)
            print('k', k.shape)
            print('v', v.shape)

        q = q.view(hs, nh, sl, bs
                  ).permute(3, 1, 2, 0).view(bs, sl*nh, hs)
        k = k.view(hs, sl, bs).permute(2, 0, 1)
        v = v.view(hs, sl, bs).permute(2, 1, 0)
        
        if print_shapes:
            print('q', q.shape)
            print('k', k.shape)
            print('v', v.shape)
        
        return q, k, v

    @classmethod
    def get_attention_weights(cls, q, k, i1, bs, sl, nh, hs, print_shapes):
        return torch.baddbmm(i1, q, k)
    
    @classmethod
    def get_attention_output(cls, attention_weights, v, i2, bs, sl, nh, hs, print_shapes):
        attn_output = torch.baddbmm(i2, attention_weights, v)
        if print_shapes:
            print('attn_output', attn_output.shape)
        #attn_output = attn_output.view(
        #    bs, sl, nh, hs).permute(1, 0, 2, 3).view(sl, bs, nh * hs)
        return attn_output
    
    
class MultiQuery1:
    '''
    bs = batch size
    sl = sequence length
    nh = number of heads
    hs = head size
    nm = number of embeddings = nh * hs
    '''

    @classmethod
    def allocate_data(cls, bs, sl, nh, hs, print_shapes):
        nm = nh * hs 
        hidden_state = torch.randn(bs, sl, nm, device=torch.device('cuda'))
        c_attn_w = torch.randn(nm, (nh + 2) * hs, device=torch.device('cuda'))
        i0 = torch.zeros(sl * bs, (nh + 2) * hs, device=torch.device('cuda'))
        i1 = torch.zeros(bs, sl * nh, sl, device=torch.device('cuda'))
        i2 = torch.zeros(bs, sl * nh, hs, device=torch.device('cuda'))
        if print_shapes:
            print('hidden_state', hidden_state.shape)
            print('c_attn_w', c_attn_w.shape)
            print('i0', i0.shape)
            print('i1', i1.shape)
            print('i2', i2.shape)
        return hidden_state, c_attn_w, i0, i1, i2

    @classmethod
    def get_qkv1(cls, hidden_state, c_attn_w, i0, bs, sl, nh, hs, print_shapes):
        return torch.addmm(
            i0,
            hidden_state,
            c_attn_w,   
        ).view(bs, sl, -1)
    
    @classmethod
    def get_qkv(cls, hidden_state, c_attn_w, i0, bs, sl, nh, hs, print_shapes):
        return torch.matmul(
            hidden_state.view(sl * bs, nh * hs),
            c_attn_w,
        ).view(bs, sl, -1)

    @classmethod
    def split_qkv(cls, qkv, bs, sl, nh, hs, print_shapes):
        q, k, v = qkv.split((nh*hs, hs, hs), dim=2)
        
        if print_shapes:
            print('q', q.shape)
            print('k', k.shape)
            print('v', v.shape)

        q = q.view(
             bs, sl, nh, hs,
        ).reshape(
            bs, sl * nh, hs
        )
        k = k.permute(0, 2, 1)
        v = v
        
        if print_shapes:
            print('q', q.shape)
            print('k', k.shape)
            print('v', v.shape)
        
        return q, k, v

    @classmethod
    def get_attention_weights(cls, q, k, i1, bs, sl, nh, hs, print_shapes):
        return torch.baddbmm(i1, q, k)
    
    @classmethod
    def get_attention_output(cls, attention_weights, v, i2, bs, sl, nh, hs, print_shapes):
        attn_output = torch.baddbmm(i2, attention_weights, v)
        if print_shapes:
            print('attn_output', attn_output.shape)
        #attn_output = attn_output.view(
        #    bs, sl, nh, hs).permute(1, 0, 2, 3).view(sl, bs, nh * hs)
        return attn_output
    
    
def get_key_totals(prof):
    names = set(('GET_QKV', 'SPLIT_QKV', 'GET_ATTENTION_WEIGHTS', 'GET_ATTENTION_OUTPUT'))
    ka = prof.key_averages()
    stats = [[el.key, el.cpu_time_total / el.count, el.cuda_time_total / el.count, el.cpu_time_total / el.count + el.cuda_time_total / el.count] for el in ka if el.key in names]
    el_total  = ['TOTAL', 0, 0, 0]
    for el in stats:
        el_total[1] += el[1]
        el_total[2] += el[2]
        el_total[3] += el[3]
    
    return [['key', 'cpu us', 'cuda us', 'all us']] + stats + [el_total]


def test_attention_total_time(cls, bs, sl, nh, hs, repeat_cnt):
    hidden_state, c_attn_w, i0, i1, i2 = cls.allocate_data(bs, sl, nh, hs, False)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i in range(repeat_cnt):
            with torch.autograd.profiler.record_function("GET_QKV"):
                qkv = cls.get_qkv(hidden_state, c_attn_w, i0, bs, sl, nh, hs, False)
            with torch.autograd.profiler.record_function("SPLIT_QKV"):
                q, k, v = cls.split_qkv(qkv, bs, sl, nh, hs, False)
            with torch.autograd.profiler.record_function("GET_ATTENTION_WEIGHTS"):
                attention_weights = cls.get_attention_weights(q, k, i1, bs, sl, nh, hs, False)
            with torch.autograd.profiler.record_function("GET_ATTENTION_OUTPUT"):
                attention_output = cls.get_attention_output(attention_weights, v, i2, bs, sl, nh, hs, False)
    res = get_key_totals(prof)
    return res

def test_attention(cls, bs, sl, nh, hs, repeat_cnt):
    hidden_state, c_attn_w, i0, i1, i2 = cls.allocate_data(bs, sl, nh, hs, True)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i in range(repeat_cnt):
            with torch.autograd.profiler.record_function("GET_QKV"):
                qkv = cls.get_qkv(hidden_state, c_attn_w, i0, bs, sl, nh, hs, i == 0)
            with torch.autograd.profiler.record_function("SPLIT_QKV"):
                q, k, v = cls.split_qkv(qkv, bs, sl, nh, hs, i == 0)
            with torch.autograd.profiler.record_function("GET_ATTENTION_WEIGHTS"):
                attention_weights = cls.get_attention_weights(q, k, i1, bs, sl, nh, hs, i == 0)
            with torch.autograd.profiler.record_function("GET_ATTENTION_OUTPUT"):
                attention_output = cls.get_attention_output(attention_weights, v, i2, bs, sl, nh, hs, i == 0)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    return get_key_totals(prof)


def test_qkv(cls, bs, sl, nh, hs, repeat_cnt):
    hidden_state, c_attn_w, i0, i1, i2 = cls.allocate_data(bs, sl, nh, hs, True)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i in range(repeat_cnt):
            with torch.autograd.profiler.record_function("GET_QKV"):
                qkv = cls.get_qkv(hidden_state, c_attn_w, i0, bs, sl, nh, hs, i == 0)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    return get_key_totals(prof)

def test_attention_weights(cls, bs, sl, nh, hs, repeat_cnt):
    hidden_state, c_attn_w, i0, i1, i2 = cls.allocate_data(bs, sl, nh, hs, True)
    qkv = cls.get_qkv(hidden_state, c_attn_w, i0, bs, sl, nh, hs, True)
    q, k, v = cls.split_qkv(qkv, bs, sl, nh, hs, True)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i in range(repeat_cnt):
            with torch.autograd.profiler.record_function("GET_ATTENTION_WEIGHTS"):
                attention_weights = cls.get_attention_weights(q, k, i1, bs, sl, nh, hs, i == 0)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    return get_key_totals(prof)

def test_attention_weights_output(cls, bs, sl, nh, hs, repeat_cnt):
    hidden_state, c_attn_w, i0, i1, i2 = cls.allocate_data(bs, sl, nh, hs, True)
    qkv = cls.get_qkv(hidden_state, c_attn_w, i0, bs, sl, nh, hs, True)
    q, k, v = cls.split_qkv(qkv, bs, sl, nh, hs, True)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i in range(repeat_cnt):
            with torch.autograd.profiler.record_function("GET_ATTENTION_WEIGHTS"):
                attention_weights = cls.get_attention_weights(q, k, i1, bs, sl, nh, hs, i == 0)
            with torch.autograd.profiler.record_function("GET_ATTENTION_OUTPUT"):
                attention_output = cls.get_attention_output(attention_weights, v, i2, bs, sl, nh, hs, i == 0)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    return get_key_totals(prof)
