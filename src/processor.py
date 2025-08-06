"""Batch processor for LM Studio text processing."""
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from client import LMStudioClient
from file_manager import FileManager
from config import Config


class BatchProcessor:
    """Orchestrates batch processing of text files through LM Studio."""
    
    def __init__(self, config: Config, verbose: bool = False):
        """Initialize batch processor.
        
        Args:
            config: Configuration object
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
        
        # Initialize components
        self.client = LMStudioClient(
            server_url=config.lm_studio['server_url'],
            timeout=config.lm_studio['timeout'],
            retry_attempts=config.lm_studio['retry_attempts'],
            retry_delay=config.lm_studio['retry_delay'],
            ctx_size=config.context_handling.get('ctx_size', 16384)
        )
        
        self.file_manager = FileManager(
            output_dir=config.output['directory']
        )
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_tokens': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate that all components are properly configured.
        
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'client_status': None,
            'file_manager_status': None,
            'errors': []
        }
        
        try:
            # Validate LM Studio connection
            client_info = self.client.validate_connection()
            results['client_status'] = client_info
            
            if not client_info['connected']:
                results['valid'] = False
                results['errors'].append(f"LM Studio connection failed: {client_info['error']}")
            
            # Validate file manager
            try:
                self.file_manager.output_dir.mkdir(exist_ok=True)
                results['file_manager_status'] = 'ready'
            except Exception as e:
                results['valid'] = False
                results['errors'].append(f"File manager setup failed: {str(e)}")
                results['file_manager_status'] = 'failed'
        
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Setup validation failed: {str(e)}")
        
        return results
    
    def process_files(self, 
                     prompt_path: str, 
                     input_paths: List[str],
                     progress_callback: Optional[Callable] = None,
                     dry_run: bool = False) -> Dict[str, Any]:
        """Process text files with the given prompt.
        
        Args:
            prompt_path: Path to prompt file
            input_paths: List of input file paths
            progress_callback: Optional callback for progress updates
            dry_run: If True, validate but don't actually process
        
        Returns:
            Processing results dictionary
        """
        self.stats['start_time'] = datetime.now()
        self.stats['total_files'] = len(input_paths)
        
        try:
            # Read prompt file
            prompt_content = self.file_manager.read_prompt_file(prompt_path)
            
            # Validate files
            validation = self.file_manager.validate_files(prompt_path, input_paths)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': 'File validation failed',
                    'details': validation['errors'],
                    'stats': self.stats
                }
            
            if dry_run:
                return {
                    'success': True,
                    'dry_run': True,
                    'validation': validation,
                    'would_process': len(input_paths),
                    'stats': self.stats
                }
            
            # Load model with specified context size
            model_name = self.config.lm_studio['model']
            ctx_size = self.config.context_handling.get('ctx_size', 16384)
            
            if self.verbose:
                print(f"Loading model '{model_name}' with context size {ctx_size}...")
            
            if not self.client.load_model_with_context(model_name, ctx_size):
                if self.verbose:
                    print(f"Warning: Failed to load model with context size {ctx_size}, continuing with current model...")
            elif self.verbose:
                print(f"Successfully loaded model with context size {ctx_size}")
            
            # Process files
            results = []
            concurrent_requests = self.config.processing['concurrent_requests']
            
            if concurrent_requests > 1:
                results = self._process_files_concurrent(
                    prompt_path, prompt_content, input_paths, concurrent_requests, progress_callback
                )
            else:
                results = self._process_files_sequential(
                    prompt_path, prompt_content, input_paths, progress_callback
                )
            
            self.stats['end_time'] = datetime.now()
            
            return {
                'success': True,
                'results': results,
                'stats': self.stats,
                'output_summary': self.file_manager.get_output_summary()
            }
        
        except Exception as e:
            self.stats['end_time'] = datetime.now()
            self.stats['errors'].append(str(e))
            
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }
    
    def _process_files_sequential(self, 
                                prompt_path: str,
                                prompt_content: str, 
                                input_paths: List[str],
                                progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Process files sequentially.
        
        Args:
            prompt_path: Path to prompt file
            prompt_content: The prompt template
            input_paths: List of input file paths
            progress_callback: Optional progress callback
        
        Returns:
            List of processing results
        """
        results = []
        
        with tqdm(total=len(input_paths), desc="Processing files", disable=not self.verbose) as pbar:
            for file_path in input_paths:
                try:
                    result = self._process_single_file(prompt_path, prompt_content, file_path)
                    results.append(result)
                    
                    if result['success']:
                        self.stats['processed_files'] += 1
                    else:
                        self.stats['failed_files'] += 1
                        self.stats['errors'].append(f"{file_path}: {result.get('error', 'Unknown error')}")
                    
                    pbar.update(1)
                    
                    if progress_callback:
                        progress_callback(len(results), len(input_paths), result)
                
                except Exception as e:
                    error_msg = f"Failed to process {file_path}: {str(e)}"
                    results.append({
                        'file_path': file_path,
                        'success': False,
                        'error': error_msg
                    })
                    self.stats['failed_files'] += 1
                    self.stats['errors'].append(error_msg)
                    pbar.update(1)
        
        return results
    
    def _process_files_concurrent(self, 
                                prompt_path: str,
                                prompt_content: str, 
                                input_paths: List[str],
                                max_workers: int,
                                progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Process files concurrently.
        
        Args:
            prompt_path: Path to prompt file
            prompt_content: The prompt template
            input_paths: List of input file paths
            max_workers: Maximum number of concurrent workers
            progress_callback: Optional progress callback
        
        Returns:
            List of processing results
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_file, prompt_path, prompt_content, file_path): file_path
                for file_path in input_paths
            }
            
            # Process completed tasks
            with tqdm(total=len(input_paths), desc="Processing files", disable=not self.verbose) as pbar:
                for future in as_completed(future_to_path):
                    file_path = future_to_path[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['success']:
                            self.stats['processed_files'] += 1
                        else:
                            self.stats['failed_files'] += 1
                            self.stats['errors'].append(f"{file_path}: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        error_msg = f"Failed to process {file_path}: {str(e)}"
                        results.append({
                            'file_path': file_path,
                            'success': False,
                            'error': error_msg
                        })
                        self.stats['failed_files'] += 1
                        self.stats['errors'].append(error_msg)
                    
                    pbar.update(1)
                    
                    if progress_callback:
                        progress_callback(len(results), len(input_paths), results[-1])
        
        return results
    
    def _process_single_file(self, prompt_path: str, prompt_content: str, file_path: str) -> Dict[str, Any]:
        """Process a single text file.
        
        Args:
            prompt_path: Path to prompt file
            prompt_content: The prompt template
            file_path: Path to the text file
        
        Returns:
            Processing result dictionary
        """
        result = {
            'file_path': file_path,
            'success': False,
            'output_files': [],
            'tokens_used': 0,
            'processing_time': 0,
            'error': None,
            'chunks_processed': 0
        }
        
        start_time = time.time()
        
        try:
            # Read text file
            text_content = self.file_manager.read_text_file(file_path)
            
            # Get context handling configuration
            model_name = self.config.lm_studio['model']
            context_config = self.config.context_handling
            
            # Determine context length
            if hasattr(self.config, '_override_context_length'):
                max_context = self.config._override_context_length
            elif context_config['auto_detect']:
                max_context = self.config.get_model_context_length(model_name)
            else:
                max_context = self.config.processing.get('max_context_length', 8192)
            
            # Combine prompt and text with strategy
            try:
                content_result = self.file_manager.combine_prompt_and_text(
                    prompt_content=prompt_content,
                    text_content=text_content,
                    max_context_length=max_context,
                    max_tokens=self.config.processing['max_tokens'],
                    strategy=context_config['strategy'],
                    safety_margin=context_config['safety_margin'],
                    warn_on_truncation=context_config['warn_on_truncation']
                )
            except ValueError as e:
                # Handle 'fail' strategy
                result.update({
                    'error': str(e),
                    'processing_time': time.time() - start_time
                })
                return result
            
            # Handle different return types from combine_prompt_and_text
            if isinstance(content_result, tuple) and len(content_result) == 2:
                # Single content (not split)
                combined_content, content_metadata = content_result
                chunks_to_process = [(combined_content, content_metadata, None)]
            elif isinstance(content_result, list):
                # Split content - list of (content, metadata) tuples
                chunks_to_process = [(content, metadata, idx+1) for idx, (content, metadata) in enumerate(content_result)]
                result['chunks_processed'] = len(chunks_to_process)
            else:
                raise ValueError("Unexpected return type from combine_prompt_and_text")
            
            # Process each chunk
            total_tokens = 0
            output_files = []
            
            for chunk_content, chunk_metadata, chunk_num in chunks_to_process:
                # Send to LM Studio
                response = self.client.send_request(
                    prompt=chunk_content,
                    model=model_name,
                    temperature=self.config.processing['temperature'],
                    max_tokens=self.config.processing['max_tokens']
                )
                
                # Extract response text
                response_text = self.client.extract_response_text(response)
                
                # Generate output filename
                output_filename = self.file_manager.generate_output_filename(
                    prompt_path, file_path, chunk_num
                )
                
                # Prepare metadata
                metadata = None
                if self.config.output['include_metadata']:
                    metadata = {
                        'processed_at': datetime.now().isoformat(),
                        'prompt_file': prompt_path,
                        'source_file': file_path,
                        'model': model_name,
                        'temperature': self.config.processing['temperature'],
                        'max_tokens': self.config.processing['max_tokens'],
                        'tokens_used': response.get('usage', {}).get('total_tokens', 0),
                        'context_length': max_context,
                        **chunk_metadata  # Include chunk-specific metadata
                    }
                
                # Write output file
                output_path = self.file_manager.write_output_file(
                    content=response_text,
                    filename=output_filename,
                    metadata=metadata,
                    overwrite=self.config.output['overwrite']
                )
                
                output_files.append(output_path)
                total_tokens += response.get('usage', {}).get('total_tokens', 0)
            
            result.update({
                'success': True,
                'output_files': output_files,
                'output_file': output_files[0] if output_files else None,  # Backwards compatibility
                'tokens_used': total_tokens,
                'processing_time': time.time() - start_time
            })
            
            self.stats['total_tokens'] += total_tokens
        
        except Exception as e:
            result.update({
                'error': str(e),
                'processing_time': time.time() - start_time
            })
        
        return result
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing results.
        
        Returns:
            Processing summary dictionary
        """
        duration = 0
        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        return {
            'total_files': self.stats['total_files'],
            'processed_files': self.stats['processed_files'],
            'failed_files': self.stats['failed_files'],
            'success_rate': self.stats['processed_files'] / max(self.stats['total_files'], 1) * 100,
            'total_tokens': self.stats['total_tokens'],
            'processing_time': duration,
            'average_time_per_file': duration / max(self.stats['processed_files'], 1),
            'errors': self.stats['errors']
        }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'client'):
            self.client.close()