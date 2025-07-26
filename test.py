import sys
import os

# Add the sts_lightspeed directory to the Python path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sts_lightspeed'))
import slaythespire

gc = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 12345, 67890)
import IPython; IPython.embed()