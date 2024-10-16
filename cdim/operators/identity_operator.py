from cdim.operators import register_operator

@register_operator(name='identity')
class IdentityOperator:
    def __init__(self, device):
        self.device = device
    
    def __call__(self, data):
        return data
