import sys
import h5py

def main(path):
    with h5py.File(path, 'r') as f:
        def rec(g, prefix=''):
            for k, v in g.items():
                if isinstance(v, h5py.Dataset):
                    print(prefix + k, v.shape, v.dtype)
                else:
                    print(prefix + k, '(group)')
                    rec(v, prefix + k + '/')
        rec(f)

if __name__ == '__main__':
    main(sys.argv[1])

