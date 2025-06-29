"""
EV-related models including vehicles, chargers, and batteries.
"""

from .vehicle import EV
from .charger import EV_Charger

__all__ = ['EV', 'EV_Charger']
