import torch

class ExperienceBuffer():
    def __init__(self, buffer_length, batch_size, device):
        self._buffer_length = buffer_length
        self._batch_size = batch_size
        self._device = device

        self._buffer_head = 0
        self._total_samples = 0

        self._buffers = dict()
        self._flat_buffers = dict()
        self._sample_buf = torch.randperm(self.get_capacity(), device=self._device, dtype=torch.long)
        self._sample_buf_head = 0
        self._reset_sample_buf()

        return

    def add_buffer(self, name, data_shape, dtype):
        assert(name not in self._buffers)

        buffer_shape = [self._buffer_length, self._batch_size] + list(data_shape)
        buffer = torch.zeros(buffer_shape, dtype=dtype, device=self._device)
        self._buffers[name] = buffer

        flat_shape = [buffer_shape[0] * buffer_shape[1]] + list(data_shape)
        self._flat_buffers[name] = buffer.view(flat_shape)
        return

    def reset(self):
        self._buffer_head = 0
        self._reset_sample_buf()
        return

    def clear(self):
        self.reset()
        self._total_samples = 0
        return

    def inc(self):
        self._buffer_head = (self._buffer_head + 1) % self._buffer_length
        self._total_samples += self._batch_size
        return

    def get_total_samples(self):
        return self._total_samples

    def get_capacity(self):
        return self._buffer_length * self._batch_size

    def get_sample_count(self):
        sample_count = min(self._total_samples, self.get_capacity())
        return sample_count

    def is_full(self):
        return self._total_samples >= self.get_capacity()

    def record(self, name, data):
        assert(data.shape[0] == self._batch_size)

        sample_count = self.get_sample_count()
        if (sample_count == 0 and name not in self._buffers):
            self.add_buffer(name, data.shape[1:], data.dtype)

        data_buf = self._buffers[name]
        data_buf[self._buffer_head] = data
        return

    def get_data(self, name):
        return self._buffers[name]

    def get_data_flat(self, name):
        return self._flat_buffers[name]
    
    def set_data(self, name, data):
        assert(data.shape[0] == self._buffer_length)
        assert(data.shape[1] == self._batch_size)
        
        if (name not in self._buffers):
            self.add_buffer(name, data.shape[2:], data.dtype)
        
        data_buf = self.get_data(name)
        data_buf[:] = data
        return
    
    def set_data_flat(self, name, data):
        assert(data.shape[0] == self._buffer_length * self._batch_size)
        
        if (name not in self._buffers):
            self.add_buffer(name, data.shape[1:], data.dtype)

        data_buf = self.get_data_flat(name)
        data_buf[:] = data
        return

    def sample(self, n):
        output = dict()
        rand_idx = self._sample_rand_idx(n)

        for key, data in self._flat_buffers.items():
            batch_data = data[rand_idx]
            output[key] = batch_data

        return output
    
    def push(self, data_dict):
        if (len(self._buffers) == 0):
            for key, data in data_dict.items():
                self.add_buffer(name=key, data_shape=data.shape[2:], dtype=data.dtype)

        n = next(iter(data_dict.values())).shape[0]
        assert(n <= self._buffer_length)

        for key, curr_buf in self._buffers.items():
            curr_data = data_dict[key]
            curr_n = curr_data.shape[0]
            curr_batch_size = curr_data.shape[1]
            assert(n == curr_n)
            assert(curr_batch_size == self._batch_size)

            store_n = min(curr_n, self._buffer_length - self._buffer_head)
            curr_buf[self._buffer_head:(self._buffer_head + store_n)] = curr_data[:store_n]    
        
            remainder = n - store_n
            if (remainder > 0):
                curr_buf[0:remainder] = curr_data[store_n:]  

        self._buffer_head = (self._buffer_head + n) % self._buffer_length
        self._total_samples += n
        return


    def _reset_sample_buf(self):
        self._sample_buf[:] = torch.randperm(self.get_capacity(), device=self._device,
                                             dtype=torch.long)
        self._sample_buf_head = 0
        return

    def _sample_rand_idx(self, n):
        buffer_len = self._sample_buf.shape[0]
        assert(n <= buffer_len)

        if (self._sample_buf_head + n <= buffer_len):
            rand_idx = self._sample_buf[self._sample_buf_head:self._sample_buf_head + n]
            self._sample_buf_head += n
        else:
            rand_idx0 = self._sample_buf[self._sample_buf_head:]
            remainder = n - (buffer_len - self._sample_buf_head)

            self._reset_sample_buf()
            rand_idx1 = self._sample_buf[:remainder]
            rand_idx = torch.cat([rand_idx0, rand_idx1], dim=0)

            self._sample_buf_head = remainder

        sample_count = self.get_sample_count()
        rand_idx = torch.remainder(rand_idx, sample_count)
        return rand_idx