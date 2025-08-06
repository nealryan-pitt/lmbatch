"""File management utilities for LM Batch processing."""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime


class FileManager:
    """Manages file operations for batch processing."""
    
    def __init__(self, output_dir: str = 'output'):
        """Initialize file manager.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def read_prompt_file(self, prompt_path: str) -> str:
        """Read and return contents of a prompt file.
        
        Args:
            prompt_path: Path to prompt file
        
        Returns:
            Contents of the prompt file
        
        Raises:
            Exception: If file cannot be read
        """
        try:
            prompt_file = Path(prompt_path)
            if not prompt_file.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
            
            if not prompt_file.is_file():
                raise ValueError(f"Prompt path is not a file: {prompt_path}")
            
            # Try different encodings
            for encoding in ['utf-8', 'utf-8-sig', 'latin1']:
                try:
                    return prompt_file.read_text(encoding=encoding).strip()
                except UnicodeDecodeError:
                    continue
            
            raise ValueError(f"Unable to decode prompt file: {prompt_path}")
        
        except Exception as e:
            raise Exception(f"Failed to read prompt file {prompt_path}: {str(e)}")
    
    def find_text_files(self, input_path: str) -> List[str]:
        """Find all text files in the given path.
        
        Args:
            input_path: Path to file or directory
        
        Returns:
            List of text file paths
        
        Raises:
            Exception: If path is invalid or inaccessible
        """
        try:
            path = Path(input_path)
            
            if not path.exists():
                raise FileNotFoundError(f"Input path not found: {input_path}")
            
            if path.is_file():
                return [str(path)]
            
            elif path.is_dir():
                # Find all text files in directory
                text_extensions = {'.txt', '.md', '.text', '.log', '.csv', '.json', '.py', '.js', '.html', '.xml'}
                text_files = []
                
                for file_path in path.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                        text_files.append(str(file_path))
                
                if not text_files:
                    raise ValueError(f"No text files found in directory: {input_path}")
                
                return sorted(text_files)
            
            else:
                raise ValueError(f"Invalid path type: {input_path}")
        
        except Exception as e:
            raise Exception(f"Failed to find text files in {input_path}: {str(e)}")
    
    def read_text_file(self, file_path: str) -> str:
        """Read and return contents of a text file.
        
        Args:
            file_path: Path to text file
        
        Returns:
            Contents of the text file
        
        Raises:
            Exception: If file cannot be read
        """
        try:
            text_file = Path(file_path)
            
            if not text_file.exists():
                raise FileNotFoundError(f"Text file not found: {file_path}")
            
            if not text_file.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            
            # Try different encodings
            for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    content = text_file.read_text(encoding=encoding)
                    return content.strip()
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, read as binary and decode with error handling
            binary_content = text_file.read_bytes()
            return binary_content.decode('utf-8', errors='replace').strip()
        
        except Exception as e:
            raise Exception(f"Failed to read text file {file_path}: {str(e)}")
    
    def combine_prompt_and_text(self, 
                                  prompt_content: str, 
                                  text_content: str, 
                                  max_context_length: int = 8192,
                                  max_tokens: int = 32000,
                                  strategy: str = 'fail',
                                  safety_margin: int = 500,
                                  warn_on_truncation: bool = True) -> tuple:
        """Combine prompt and text content for processing.
        
        Args:
            prompt_content: The prompt template
            text_content: The text to process
            max_context_length: Maximum context length in tokens
            max_tokens: Maximum tokens for response
            strategy: How to handle oversized content ('fail', 'truncate', 'split', 'force')
            safety_margin: Tokens to reserve as safety buffer
            warn_on_truncation: Whether to warn when truncating
        
        Returns:
            Tuple of (combined_content, metadata_dict)
            For split strategy, returns list of (chunk, metadata) tuples
        
        Raises:
            ValueError: If content is too large and strategy is 'fail'
        """
        separator = "\n\n---\n\n"
        
        # Estimate tokens (rough approximation: ~4 chars per token)
        prompt_tokens = len(prompt_content) // 4
        separator_tokens = len(separator) // 4
        text_tokens = len(text_content) // 4
        total_tokens = prompt_tokens + separator_tokens + text_tokens
        
        # Calculate available space
        response_buffer = min(max_tokens, 2048)  # Reserve reasonable space for response
        available_tokens = max_context_length - prompt_tokens - separator_tokens - safety_margin - response_buffer
        
        metadata = {
            'prompt_tokens': prompt_tokens,
            'text_tokens': text_tokens,
            'total_input_tokens': total_tokens,
            'context_limit': max_context_length,
            'available_tokens': available_tokens,
            'strategy_used': strategy,
            'was_truncated': False,
            'was_split': False,
        }
        
        # Check if content fits
        if text_tokens <= available_tokens:
            # Content fits, no modification needed
            combined = f"{prompt_content}{separator}{text_content}"
            return combined, metadata
        
        # Content is too large, apply strategy
        if strategy == 'fail':
            raise ValueError(
                f"Content too large for context window:\n"
                f"  Prompt: {prompt_tokens:,} tokens\n"
                f"  Text: {text_tokens:,} tokens\n" 
                f"  Total: {total_tokens:,} tokens\n"
                f"  Context limit: {max_context_length:,} tokens\n"
                f"  Available for text: {available_tokens:,} tokens\n\n"
                f"Suggested solutions:\n"
                f"  1. Use larger context model: --model llama-3.3-70b\n"
                f"  2. Split processing: --strategy split\n"
                f"  3. Force send anyway: --strategy force\n"
                f"  4. Allow truncation: --strategy truncate"
            )
        
        elif strategy == 'force':
            # Send anyway, let LM Studio handle the overflow
            combined = f"{prompt_content}{separator}{text_content}"
            metadata['strategy_used'] = 'force'
            return combined, metadata
        
        elif strategy == 'truncate':
            # Truncate text content
            max_text_chars = available_tokens * 4
            if len(text_content) > max_text_chars:
                # Truncate at word boundary
                truncated_text = text_content[:max_text_chars].rsplit(' ', 1)[0]
                text_content = f"{truncated_text}\n\n[NOTE: Text was truncated due to context length limits - {len(text_content) - len(truncated_text)} characters removed]"
                metadata['was_truncated'] = True
                metadata['truncated_chars'] = len(text_content) - len(truncated_text)
                
                if warn_on_truncation:
                    print(f"⚠️  WARNING: Text truncated by {metadata['truncated_chars']} characters to fit context window")
            
            combined = f"{prompt_content}{separator}{text_content}"
            return combined, metadata
        
        elif strategy == 'split':
            # Split into chunks - return list of (chunk, metadata) tuples
            return self._split_content(prompt_content, text_content, available_tokens, metadata)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Must be one of: fail, truncate, split, force")
    
    def _split_content(self, prompt_content: str, text_content: str, available_tokens: int, base_metadata: dict) -> list:
        """Split content into processable chunks.
        
        Args:
            prompt_content: The prompt template
            text_content: The text to split
            available_tokens: Available tokens per chunk
            base_metadata: Base metadata to extend
        
        Returns:
            List of (combined_content, metadata) tuples
        """
        separator = "\n\n---\n\n"
        overlap_chars = base_metadata.get('overlap_tokens', 300) * 4  # Convert tokens to chars
        
        # Calculate chunk size in characters
        chunk_size_chars = available_tokens * 4
        
        # Split text into chunks with overlap
        chunks = []
        text_length = len(text_content)
        start = 0
        chunk_num = 1
        
        while start < text_length:
            # Calculate end position
            end = min(start + chunk_size_chars, text_length)
            
            # If not the last chunk, try to break at word boundary
            if end < text_length:
                # Find last space within reasonable distance
                space_pos = text_content.rfind(' ', start, end)
                if space_pos > start + chunk_size_chars * 0.8:  # Don't break too early
                    end = space_pos
            
            # Extract chunk
            chunk_text = text_content[start:end]
            
            # Add overlap from previous chunk (except for first chunk)
            if start > 0:
                overlap_start = max(0, start - overlap_chars)
                overlap_text = text_content[overlap_start:start]
                chunk_text = f"[...continued from previous chunk]\n{overlap_text}\n---\n{chunk_text}"
            
            # Add continuation indicator if not last chunk
            if end < text_length:
                chunk_text += "\n[...continues in next chunk]"
            
            # Combine with prompt
            combined = f"{prompt_content}{separator}[CHUNK {chunk_num} of estimated {(text_length // chunk_size_chars) + 1}]\n\n{chunk_text}"
            
            # Create chunk metadata
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'was_split': True,
                'chunk_number': chunk_num,
                'chunk_start': start,
                'chunk_end': end,
                'chunk_chars': len(chunk_text),
                'chunk_tokens': len(chunk_text) // 4,
            })
            
            chunks.append((combined, chunk_metadata))
            
            # Move to next chunk (with overlap consideration)
            start = end
            chunk_num += 1
        
        return chunks
    
    def generate_output_filename(self, prompt_path: str, text_path: str, chunk_number: int = None) -> str:
        """Generate output filename based on input files.
        
        Args:
            prompt_path: Path to prompt file
            text_path: Path to text file
            chunk_number: Chunk number for split processing
        
        Returns:
            Generated output filename
        """
        prompt_name = Path(prompt_path).stem
        text_name = Path(text_path).stem
        
        if chunk_number is not None:
            return f"{prompt_name}.{text_name}.chunk{chunk_number}.txt"
        else:
            return f"{prompt_name}.{text_name}.txt"
    
    def write_output_file(self, 
                         content: str, 
                         filename: str, 
                         metadata: Optional[Dict[str, Any]] = None,
                         overwrite: bool = False) -> str:
        """Write content to output file.
        
        Args:
            content: Content to write
            filename: Output filename
            metadata: Optional metadata to include
            overwrite: Whether to overwrite existing files
        
        Returns:
            Path to written file
        
        Raises:
            Exception: If file cannot be written
        """
        try:
            output_path = self.output_dir / filename
            
            if output_path.exists() and not overwrite:
                # Generate unique filename
                base = output_path.stem
                suffix = output_path.suffix
                counter = 1
                
                while output_path.exists():
                    new_name = f"{base}_{counter:03d}{suffix}"
                    output_path = self.output_dir / new_name
                    counter += 1
            
            # Prepare content with metadata if requested
            final_content = content
            if metadata:
                metadata_header = self._format_metadata(metadata)
                final_content = f"{metadata_header}\n\n{content}"
            
            # Write file
            output_path.write_text(final_content, encoding='utf-8')
            return str(output_path)
        
        except Exception as e:
            raise Exception(f"Failed to write output file {filename}: {str(e)}")
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata as a header comment.
        
        Args:
            metadata: Metadata dictionary
        
        Returns:
            Formatted metadata header
        """
        lines = ["<!-- LM Batch Processing Metadata"]
        
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, indent=2)
            lines.append(f"{key}: {value}")
        
        lines.append("-->")
        return "\n".join(lines)
    
    def validate_files(self, prompt_path: str, text_paths: List[str]) -> Dict[str, Any]:
        """Validate all input files before processing.
        
        Args:
            prompt_path: Path to prompt file
            text_paths: List of text file paths
        
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_count': len(text_paths),
            'total_size': 0
        }
        
        # Validate prompt file
        try:
            prompt_content = self.read_prompt_file(prompt_path)
            if not prompt_content:
                results['warnings'].append(f"Prompt file is empty: {prompt_path}")
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Prompt file error: {str(e)}")
        
        # Validate text files
        for text_path in text_paths:
            try:
                content = self.read_text_file(text_path)
                results['total_size'] += len(content.encode('utf-8'))
                
                if not content:
                    results['warnings'].append(f"Text file is empty: {text_path}")
                
            except Exception as e:
                results['errors'].append(f"Text file error ({text_path}): {str(e)}")
        
        # Check output directory
        if not os.access(self.output_dir, os.W_OK):
            results['valid'] = False
            results['errors'].append(f"Output directory is not writable: {self.output_dir}")
        
        if results['errors']:
            results['valid'] = False
        
        return results
    
    def cleanup_output_dir(self, pattern: str = None):
        """Clean up output directory.
        
        Args:
            pattern: Optional glob pattern to match files for deletion
        """
        try:
            if pattern:
                for file_path in self.output_dir.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
            else:
                for file_path in self.output_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
        except Exception as e:
            raise Exception(f"Failed to cleanup output directory: {str(e)}")
    
    def get_output_summary(self) -> Dict[str, Any]:
        """Get summary of output directory.
        
        Returns:
            Summary dictionary with file counts and sizes
        """
        try:
            files = list(self.output_dir.glob('*.txt'))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            return {
                'output_dir': str(self.output_dir),
                'file_count': len(files),
                'total_size': total_size,
                'files': [str(f.name) for f in files]
            }
        except Exception as e:
            return {'error': str(e)}