from .dsgg_single_sgdet import build as build_single_sgdet
from .dsgg_single_xcls import build as build_single_xcls
from .dsgg_multi_sgdet import build as build_multi_sgdet
from .dsgg_multi_xcls import build as build_multi_xcls
# from .dsgg_single_xcls_one_dec import build as buidl_single_predcls_one_dec
from .dsgg_multi_2 import build as build_multi_2

def build_model(args):
    if args.dataset_file == 'ag_single':
        if args.dsgg_task == 'sgdet':
            return build_single_sgdet(args)
        else:
            # if args.one_dec and args.dsgg_task == 'predcls':
            #     return buidl_single_predcls_one_dec(args)
            # else:
            return build_single_xcls(args)
    elif args.dataset_file == 'ag_multi':
        if args.dsgg_task == 'sgdet':
            if args.method2: # temp trying
                return build_multi_2(args)
            return build_multi_sgdet(args)
        else:
            return build_multi_xcls(args)
    else:
        raise NotImplementedError