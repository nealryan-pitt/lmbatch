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
    
    def combine_prompt_and_text(self, prompt_content: str, text_content: str, max_context_length: int = 3000) -> str:
        """Combine prompt and text content for processing.
        
        Args:
            prompt_content: The prompt template
            text_content: The text to process
            max_context_length: Maximum context length in tokens (approximate)
        
        Returns:
            Combined content ready for LLM
        """
        separator = "\n\n---\n\n"
        
        # Estimate tokens (rough approximation: ~4 chars per token)
        prompt_tokens = len(prompt_content) // 4
        separator_tokens = len(separator) // 4
        available_tokens = max_context_length - prompt_tokens - separator_tokens - 200  # Reserve 200 tokens for response
        
        if available_tokens <= 0:
            raise ValueError("Prompt is too long for the specified context length")
        
        # Truncate text content if necessary
        max_text_chars = available_tokens * 4
        if len(text_content) > max_text_chars:
            # Truncate and add indication
            truncated_text = text_content[:max_text_chars].rsplit(' ', 1)[0]  # Don't cut words
            text_content = f"{truncated_text}\n\n[NOTE: Text was truncated due to context length limits]"
        
        return f"{prompt_content}{separator}{text_content}"
    
    def generate_output_filename(self, prompt_path: str, text_path: str) -> str:
        """Generate output filename based on input files.
        
        Args:
            prompt_path: Path to prompt file
            text_path: Path to text file
        
        Returns:
            Generated output filename
        """
        prompt_name = Path(prompt_path).stem
        text_name = Path(text_path).stem
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