from ablkit.utils import print_log
from ablkit.reasoning import KBBase


class AddKB(KBBase):
    def __init__(self, pseudo_label_list=list(range(10))):
        super().__init__(pseudo_label_list)

    # Implement the deduction function
    def logic_forward(self, nums):
        return sum(nums)
    

def test(kb, pseudo_labels):    
    reasoning_result = kb.logic_forward(pseudo_labels)
    print_log(f"Reasoning result of pseudo-labels {pseudo_labels} is {reasoning_result}.", 
                logger="current"
    )
