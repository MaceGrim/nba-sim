import torch
from transformers import PreTrainedTokenizerFast
from train_model_focused_context import (
    create_focused_attention_mask,
    extract_game_state,
    group_texts,
    BLOCK_SIZE
)
import unittest
import numpy as np

class TestFocusedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that will be used for all tests"""
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_file="custom_whitespace_tokenizer.json")
        
        # Sample game sequence
        cls.sample_sequence = """
BOS Marcus_Smart SubOut Derrick_White SubIn <EVENT_END>
Q1 0:02:51 22 17 Home PHI OnCourtAway Danuel_House_Jr_ De_Anthony_Melton Georges_Niang James_Harden Joel_Embiid Matisse_Thybulle Montrezl_Harrell P_J__Tucker Tobias_Harris Tyrese_Maxey BOS OnCourtHome Al_Horford Blake_Griffin Derrick_White Grant_Williams Jaylen_Brown Jayson_Tatum Malcolm_Brogdon Marcus_Smart Noah_Vonleh Sam_Hauser
PHI Tobias_Harris SubOut Georges_Niang SubIn <EVENT_END>
Q1 0:02:51 22 17 Home PHI OnCourtAway Danuel_House_Jr_ De_Anthony_Melton Georges_Niang James_Harden Joel_Embiid Matisse_Thybulle Montrezl_Harrell P_J__Tucker Tobias_Harris Tyrese_Maxey BOS OnCourtHome Al_Horford Blake_Griffin Derrick_White Grant_Williams Jaylen_Brown Jayson_Tatum Malcolm_Brogdon Marcus_Smart Noah_Vonleh Sam_Hauser
PHI James_Harden FreeThrow 1 1 Made <EVENT_END>
""".strip()

    def test_extract_game_state(self):
        """Test that game state extraction works correctly"""
        lines = self.sample_sequence.split('\n')
        game_state = extract_game_state(lines[1])
        
        self.assertIn("Q1", game_state)
        self.assertIn("0:02:51", game_state)
        self.assertIn("22 17", game_state)
        self.assertIn("OnCourtAway", game_state)
        self.assertIn("OnCourtHome", game_state)

    def test_attention_mask_creation(self):
        """Test that attention masks are created correctly"""
        # Tokenize a sequence
        tokens = self.tokenizer(self.sample_sequence, return_tensors="pt")
        input_ids = tokens["input_ids"][0]
        
        attention_mask = create_focused_attention_mask(input_ids, self.tokenizer)
        
        # Check mask shape
        self.assertEqual(attention_mask.shape, input_ids.shape)
        
        # Find EVENT_END tokens
        event_end_token = self.tokenizer.convert_tokens_to_ids('<EVENT_END>')
        event_ends = (input_ids == event_end_token).nonzero(as_tuple=True)[0]
        
        # Check that recent events have higher attention
        if len(event_ends) >= 2:
            last_event_start = event_ends[-2] + 1
            # Check attention values for recent events
            self.assertTrue(torch.all(attention_mask[last_event_start:] > 1.0))
            
            # Check attention values for older events
            if len(event_ends) > 2:
                self.assertTrue(torch.all(attention_mask[:event_ends[-3]] < 1.0))

    def test_block_size_compliance(self):
        """Test that all generated chunks comply with BLOCK_SIZE"""
        # Tokenize the sequence
        tokens = self.tokenizer(
            {"text": [self.sample_sequence]},
            truncation=False,
            add_special_tokens=False,
            return_token_type_ids=False
        )
        
        # Group into chunks
        chunks = group_texts(tokens)
        
        # Check all chunks
        for chunk in chunks["input_ids"]:
            self.assertLessEqual(len(chunk), BLOCK_SIZE)

    def test_event_boundaries(self):
        """Test that events are not split across chunks"""
        # Tokenize the sequence
        tokens = self.tokenizer(
            {"text": [self.sample_sequence]},
            truncation=False,
            add_special_tokens=False,
            return_token_type_ids=False
        )
        
        # Group into chunks
        chunks = group_texts(tokens)
        
        event_end_token = self.tokenizer.convert_tokens_to_ids('<EVENT_END>')
        
        # Check each chunk ends with EVENT_END
        for chunk in chunks["input_ids"]:
            # Find all EVENT_END tokens in chunk
            event_ends = [i for i, token in enumerate(chunk) 
                         if token == event_end_token and i < BLOCK_SIZE]
            
            if event_ends:  # If chunk contains any events
                # Last non-padding token should be EVENT_END
                last_real_token_pos = max(i for i, token in enumerate(chunk) 
                                        if token != self.tokenizer.pad_token_id)
                self.assertEqual(chunk[last_real_token_pos], event_end_token)

    def test_context_preservation(self):
        """Test that each chunk preserves necessary context"""
        # Tokenize the sequence
        tokens = self.tokenizer(
            {"text": [self.sample_sequence]},
            truncation=False,
            add_special_tokens=False,
            return_token_type_ids=False
        )
        
        # Group into chunks
        chunks = group_texts(tokens)
        
        for chunk_ids in chunks["input_ids"]:
            # Decode chunk back to text
            chunk_text = self.tokenizer.decode(chunk_ids)
            
            # Count events in chunk
            event_count = chunk_text.count('<EVENT_END>')
            
            # Each chunk should have at least 1 event (except maybe the last chunk)
            if not all(t == self.tokenizer.pad_token_id for t in chunk_ids):
                self.assertGreaterEqual(event_count, 1)
                
            # Check for game state information
            if event_count > 0:
                self.assertIn("Q1", chunk_text)  # Should contain quarter info
                self.assertIn("OnCourt", chunk_text)  # Should contain player lists

def main():
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main()