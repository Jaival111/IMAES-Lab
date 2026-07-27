frames = {
    "animal": {"has": "skin"},
    "bird": {"ako": "animal", "can": "fly"},
    "penguin": {"ako": "bird", "cannot": "fly"},
    "ostrich": {"ako": "bird", "cannot": "fly", "can": "run fast"}
}

def frame_query(concept, slot, frames_db):
    current = concept
    while current:
        concept_frame = frames_db.get(current, {})
        
        # Check for exception
        if slot == "can" and "cannot" in concept_frame:
            return f"cannot {concept_frame['cannot']}"
            
        # Check for local value
        if slot in concept_frame:
            return concept_frame[slot]
            
        # Move up the 'ako' chain
        current = concept_frame.get("ako")
        
    return "unknown"

print(f"Frame Query 'ostrich can': {frame_query('ostrich', 'can', frames)}")