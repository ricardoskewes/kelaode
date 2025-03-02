"""
Utility functions for JSON serialization.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy types, Pandas objects, and other non-serializable types.
    """
    def default(self, obj):
        # Handle NumPy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle Pandas objects
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        
        # Handle datetime objects
        elif isinstance(obj, datetime):
            return obj.isoformat()
        
        # Let the base class handle other types or raise TypeError
        try:
            return super(CustomJSONEncoder, self).default(obj)
        except TypeError:
            # For any other non-serializable types, convert to string
            return str(obj)

def save_json(data, filename, indent=2):
    """
    Save data to a JSON file using the custom encoder.
    
    Args:
        data: Data to save
        filename: Filename to save to
        indent: Indentation level for JSON formatting
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=indent, cls=CustomJSONEncoder)
    
    print(f"Data saved to {filename}")
    return filename
