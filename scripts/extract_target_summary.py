"""
Extract and summarize emissions reduction targets from search results
"""
import os
import sys
import re
import pandas as pd
import ast
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from the climate_policy_extractor package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from climate_policy_extractor.logging import get_logger

logger = get_logger(__name__)

def extract_target_values(text):
    """
    Extract potential emissions target values from text.
    Includes direction symbols like "-" to indicate increases/decreases.
    """
    if not text or isinstance(text, str) and len(text) < 10:  # Using 10 as a minimum length threshold
        return {
            'has_target': False,
            'percentage': None,
            'baseline_year': None,
            'target_year': None,
            'conditional': None,
            'target_type': None
        }
    
    result = {
        'has_target': False,
        'percentage': None,
        'baseline_year': None,
        'target_year': None,
        'conditional': None,
        'target_type': None
    }
    
    # Look for percentage patterns, including direction indicators
    percentage_patterns = [
        r'(-?\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)%',  # e.g., 30%, -30% or 30-40%
        r'(-?\d+(?:\.\d+)?) percent',               # e.g., 30 percent, -30 percent
        r'reduce.*?by (\d+(?:\.\d+)?)%',            # e.g., reduce... by 30%
        r'reduction of (\d+(?:\.\d+)?)%',           # e.g., reduction of 30%
        r'increase.*?by (\d+(?:\.\d+)?)%',          # e.g., increase... by 30%
        r'increment of (\d+(?:\.\d+)?)%'            # e.g., increment of 30%
    ]
    
    for pattern in percentage_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            result['has_target'] = True
            # For patterns with "increase", prepend a "-" if it doesn't already have one
            if "increase" in pattern and matches[0] and not matches[0].startswith('-'):
                result['percentage'] = f"-{matches[0]}"
            else:
                result['percentage'] = matches[0]
            break
    
    # Look for baseline years
    baseline_patterns = [
        r'from (\d{4}) levels',
        r'compared to (\d{4})',
        r'relative to (\d{4})',
        r'base year (\d{4})',
        r'baseline (\d{4})'
    ]
    
    for pattern in baseline_patterns:
        matches = re.search(pattern, text.lower())
        if matches:
            result['baseline_year'] = matches.group(1)
            break
    
    # Look for target years
    target_year_patterns = [
        r'by (\d{4})',
        r'until (\d{4})',
        r'in (\d{4})',
        r'for (\d{4})'
    ]
    
    for pattern in target_year_patterns:
        matches = re.search(pattern, text.lower())
        if matches and matches.group(1) in ['2020', '2025', '2030', '2035', '2040', '2045', '2050']:
            result['target_year'] = matches.group(1)
            break
    
    # Fallback to specific year mentions
    if not result['target_year']:
        if '2030' in text:
            result['target_year'] = '2030'
        elif '2025' in text:
            result['target_year'] = '2025'
        elif '2050' in text:
            result['target_year'] = '2050'
    
    # Check if conditional or unconditional
    if 'conditional' in text.lower():
        result['conditional'] = True
    elif 'unconditional' in text.lower():
        result['conditional'] = False
    
    # Determine target type
    target_types = {
        'ghg': ['ghg', 'greenhouse gas', 'emission'],
        'carbon': ['carbon', 'co2', 'carbon dioxide'],
        'energy': ['energy', 'renewable'],
        'intensity': ['intensity', 'per gdp', 'per capita']
    }
    
    for type_name, keywords in target_types.items():
        if any(keyword in text.lower() for keyword in keywords):
            result['target_type'] = type_name
            break
    
    return result

def load_search_results():
    """
    Load pre-computed search results from CSV files.
    Returns a combined DataFrame with search results.
    """
    results_dir = os.path.join('data', 'results')
    similarity_path = os.path.join(results_dir, 'similarity_search_results.csv')
    word2vec_path = os.path.join(results_dir, 'word2vec_search_results.csv')
    
    # Check if files exist
    similarity_exists = os.path.exists(similarity_path)
    word2vec_exists = os.path.exists(word2vec_path)
    
    if not similarity_exists and not word2vec_exists:
        logger.error("Both search result files are missing!")
        print("ERROR: No search result files found.")
        print("Please run emissions_target_search.py to generate similarity_search_results.csv")
        print("and run Notebook 03 to generate word2vec_search_results.csv")
        return None
    
    all_results = []
    
    # Load similarity search results if available
    if (similarity_exists):
        try:
            similarity_df = pd.read_csv(similarity_path)
            if not similarity_df.empty:
                logger.info(f"Loaded {len(similarity_df)} results from similarity search")
                all_results.append(similarity_df)
            else:
                logger.warning("Similarity search results file exists but is empty")
        except Exception as e:
            logger.error(f"Error loading similarity search results: {e}")
            print(f"Error loading similarity search results: {e}")
    else:
        logger.warning("Similarity search results file not found")
        print("WARNING: similarity_search_results.csv not found.")
        print("Run emissions_target_search.py to generate this file.")
    
    # Load word2vec search results if available
    if word2vec_exists:
        try:
            word2vec_df = pd.read_csv(word2vec_path)
            if not word2vec_df.empty:
                logger.info(f"Loaded {len(word2vec_df)} results from word2vec search")
                all_results.append(word2vec_df)
            else:
                logger.warning("Word2vec search results file exists but is empty")
        except Exception as e:
            logger.error(f"Error loading word2vec search results: {e}")
            print(f"Error loading word2vec search results: {e}")
    else:
        logger.warning("Word2vec search results file not found")
        print("WARNING: word2vec_search_results.csv not found.")
        print("Run Notebook 03 to generate this file.")
    
    if not all_results:
        return None
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    logger.info(f"Combined results: {len(combined_df)} entries from {combined_df['country'].nunique()} countries")
    
    # Remove duplicates (same content from same country)
    combined_df = combined_df.drop_duplicates(subset=['country', 'content'])
    logger.info(f"After removing duplicates: {len(combined_df)} entries")
    
    # Sort by total_score within each country
    combined_df = combined_df.sort_values(['country', 'total_score'], ascending=[True, False])
    
    return combined_df

def load_validation_data():
    """
    Load validation data from the similarity_search_results.csv file.
    Returns a DataFrame with parsed emissions targets information.
    """
    csv_path = os.path.join('data', 'results', 'similarity_search_results.csv')
    try:
        if not os.path.exists(csv_path):
            logger.warning(f"Validation file {csv_path} not found.")
            return None
        
        df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning(f"Validation file {csv_path} exists but is empty.")
            return None
        
        logger.info(f"Loaded validation data from {csv_path}: {len(df)} entries")
        
        # Process the content column to extract target values
        if 'content' in df.columns:
            logger.info("Extracting target values from content in validation data")
            # Create a new dataframe with extracted values
            extracted_data = []
            
            # Parse each content entry
            for _, row in df.iterrows():
                target_info = extract_target_values(row['content'])
                target_info['country'] = row['country']
                target_info['source_content'] = row['content']
                extracted_data.append(target_info)
            
            # Convert to dataframe
            validation_df = pd.DataFrame(extracted_data)
            logger.info(f"Extracted target values from {len(validation_df)} validation entries")
            return validation_df
        
        return df
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        return None

def validate_target(country_summary, validation_df):
    """
    Validate the target extracted from the database against the validation CSV.
    Increases confidence if both sources align.
    Returns a validation boost score and whether there was a match.
    """
    if validation_df is None or validation_df.empty:
        return 0.0, False
    
    country = country_summary['country']
    
    # Filter validation data for the specific country
    country_validation = validation_df[validation_df['country'].str.lower() == country.lower()]
    
    if country_validation.empty:
        return 0.0, False
    
    validation_boost = 0.0
    has_match = False
    
    # Compare target percentage with parsed values
    if country_summary['target_percentage'] is not None and 'percentage' in country_validation.columns:
        extracted_percentage = str(country_summary['target_percentage'])
        for idx, row in country_validation.iterrows():
            if pd.notnull(row.get('percentage')):
                csv_pct = str(row['percentage'])
                if extracted_percentage == csv_pct or extracted_percentage in csv_pct:
                    validation_boost += 0.3
                    has_match = True
                    # Extract page number and paragraph id from the matching row
                    country_summary['page_number'] = row.get('page_number')
                    country_summary['paragraph_id'] = row.get('paragraph_id')
                    break
    
    # Compare target year - fixed to explicitly check for None and use exact matching
    if not has_match and country_summary['target_year'] is not None and 'target_year' in country_validation.columns:
        for idx, row in country_validation.iterrows():
            if pd.notnull(row.get('target_year')):
                if str(country_summary['target_year']) == str(row['target_year']):
                    validation_boost += 0.2
                    has_match = True
                    # Extract page number and paragraph id from the matching row
                    country_summary['page_number'] = row.get('page_number')
                    country_summary['paragraph_id'] = row.get('paragraph_id')
                    break
    
    return validation_boost, has_match

def assign_confidence_band(confidence):
    """
    Assign a confidence band based on the confidence score.
    """
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"

def parse_chunk_metadata(chunk_metadata):
    """
    Parse chunk metadata to extract page_number, element_type, and paragraph_id.
    """
    # Handle string representation of dict
    if isinstance(chunk_metadata, str):
        try:
            # Try to parse as a dict first
            chunk_metadata = ast.literal_eval(chunk_metadata)
        except (ValueError, SyntaxError):
            try:
                # Try to parse as JSON if ast.literal_eval fails
                chunk_metadata = json.loads(chunk_metadata)
            except json.JSONDecodeError:
                return {"page_number": None, "element_type": None, "paragraph_id": None}

    if not isinstance(chunk_metadata, dict):
        return {"page_number": None, "element_type": None, "paragraph_id": None}

    # Extract page number
    page_number = chunk_metadata.get('page_number')
    
    # Extract element type
    element_type = None
    if 'element_type' in chunk_metadata:
        element_type = chunk_metadata['element_type']
    elif 'element_types' in chunk_metadata and chunk_metadata['element_types']:
        if isinstance(chunk_metadata['element_types'], list):
            element_type = chunk_metadata['element_types'][0]
        else:
            element_type = chunk_metadata['element_types']
    
    # Extract paragraph_id
    paragraph_id = None
    if 'paragraph_id' in chunk_metadata:
        paragraph_id = chunk_metadata['paragraph_id']
    elif 'paragraph_ids' in chunk_metadata and chunk_metadata['paragraph_ids']:
        if isinstance(chunk_metadata['paragraph_ids'], list) and len(chunk_metadata['paragraph_ids']) > 0:
            paragraph_id = chunk_metadata['paragraph_ids'][0]
        else:
            paragraph_id = chunk_metadata['paragraph_ids']
    
    return {"page_number": page_number, "element_type": element_type, "paragraph_id": paragraph_id}

def summarize_targets(df, validation_df=None):
    """
    Analyze search results and extract structured information about emissions targets.
    Returns a new DataFrame with summarized target information by country.
    """
    if df.empty:
        return pd.DataFrame()
    
    summaries = []
    
    # Group by country
    for country, group in df.groupby('country'):
        # Initialize country summary
        country_summary = {
            'country': country,
            'has_clear_target': False,
            'target_percentage': None,
            'baseline_year': None,
            'target_year': None,
            'conditional': None,
            'target_type': None,
            'confidence': 0.0,
            'confidence_band': 'LOW',
            'validation_match': False,
            'source_text': None,
            'page_number': None,
            'paragraph_id': None,
            'doc_id': None
        }
        
        # Go through each result for this country
        max_confidence = 0.0
        best_result = None
        
        for _, row in group.iterrows():
            # Extract target information
            target_info = extract_target_values(row['content'])
            
            # Calculate confidence based on score and extracted information
            confidence = row['total_score']
            
            # Boost confidence if we found specific target information
            if target_info['has_target']:
                confidence += 0.2
            if target_info['baseline_year']:
                confidence += 0.1
            if target_info['target_year']:
                confidence += 0.1
            if target_info['conditional'] is not None:
                confidence += 0.05
            if target_info['target_type']:
                confidence += 0.05
            
            # Extract metadata (page number, paragraph ID, etc.)
            # First try chunk_metadata
            metadata = {}
            if 'chunk_metadata' in row:
                metadata = parse_chunk_metadata(row['chunk_metadata'])
            # Direct fields override metadata from chunk_metadata
            if 'page_number' in row and row['page_number'] is not None:
                metadata['page_number'] = row['page_number']
            if 'paragraph_id' in row and row['paragraph_id'] is not None:
                metadata['paragraph_id'] = row['paragraph_id']
            if 'element_type' in row and row['element_type'] is not None:
                metadata['element_type'] = row['element_type']
            
            # Check if target is validated by external data
            if target_info['has_target']:
                temp_summary = country_summary.copy()
                temp_summary['target_percentage'] = target_info['percentage']
                temp_summary['baseline_year'] = target_info['baseline_year']
                temp_summary['target_year'] = target_info['target_year']
                temp_summary['conditional'] = target_info['conditional']
                temp_summary['target_type'] = target_info['target_type']
                
                validation_boost, is_validated = validate_target(temp_summary, validation_df)
                confidence += validation_boost
                
                # Keep the result with highest confidence
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_result = row
                    country_summary['has_clear_target'] = True
                    country_summary['target_percentage'] = target_info['percentage']
                    country_summary['baseline_year'] = target_info['baseline_year']
                    country_summary['target_year'] = target_info['target_year']
                    country_summary['conditional'] = target_info['conditional']
                    country_summary['target_type'] = target_info['target_type']
                    country_summary['confidence'] = confidence
                    country_summary['validation_match'] = is_validated
                    country_summary['source_text'] = row['content']
                    country_summary['doc_id'] = row['doc_id'] if 'doc_id' in row else None
                    country_summary['page_number'] = metadata.get('page_number')
                    country_summary['paragraph_id'] = metadata.get('paragraph_id')
        
        # If no clear target was found but we have results, use the top result anyway
        if not country_summary['has_clear_target'] and not group.empty:
            top_row = group.iloc[0]
            country_summary['source_text'] = top_row['content']
            metadata = {}
            if 'chunk_metadata' in top_row:
                metadata = parse_chunk_metadata(top_row['chunk_metadata'])
            # Direct fields override metadata from chunk_metadata
            if 'page_number' in top_row and top_row['page_number'] is not None:
                metadata['page_number'] = top_row['page_number']
            if 'paragraph_id' in top_row and top_row['paragraph_id'] is not None:
                metadata['paragraph_id'] = top_row['paragraph_id']
            
            country_summary['page_number'] = metadata.get('page_number')
            country_summary['paragraph_id'] = metadata.get('paragraph_id')
            country_summary['doc_id'] = top_row['doc_id'] if 'doc_id' in top_row else None
            country_summary['confidence'] = top_row['total_score'] if 'total_score' in top_row else 0.5
        
        # Assign confidence band
        country_summary['confidence_band'] = assign_confidence_band(country_summary['confidence'])
        
        summaries.append(country_summary)
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summaries)
    
    # Sort by country
    summary_df = summary_df.sort_values('country')
    
    return summary_df

def format_summary(df):
    """Format the summary for display"""
    if df.empty:
        return "No summary results found."
    
    results = []
    
    results.append("\n=== EMISSIONS REDUCTION TARGETS BY COUNTRY ===\n")
    
    for _, row in df.iterrows():
        results.append(f"{'='*80}")
        results.append(f"COUNTRY: {row['country'].upper()}")
        
        if row['has_clear_target']:
            # Check if the target percentage starts with a negative sign
            if row['target_percentage'] and str(row['target_percentage']).startswith('-'):
                # Remove the negative sign and indicate it's an increase
                target_value = str(row['target_percentage']).lstrip('-')
                target_description = f"Target: {target_value}% increase"
            else:
                target_description = f"Target: {row['target_percentage']}% reduction"
            
            if row['target_year']:
                target_description += f" by {row['target_year']}"
            
            if row['baseline_year']:
                target_description += f" from {row['baseline_year']} levels"
            
            if row['conditional'] == True:
                target_description += " (CONDITIONAL)"
            elif row['conditional'] == False:
                target_description += " (UNCONDITIONAL)"
                
            if row['target_type']:
                target_description += f" - Type: {row['target_type'].upper()}"
                
            results.append(target_description)
            results.append(f"Confidence: {row['confidence']:.2f} ({row['confidence_band']})")
            if row['validation_match']:
                results.append("✓ Validated by external data")
        else:
            results.append("No clear target identified")
        
        source_info = f"Source: {row['doc_id']}"
        if pd.notnull(row['page_number']):
            source_info += f" (Page {row['page_number']}"
            if pd.notnull(row['paragraph_id']):
                source_info += f", Paragraph {row['paragraph_id']}"
            source_info += ")"
        
        results.append(source_info)
        results.append(f"Text: {row['source_text']}")
        results.append("")
    
    return "\n".join(results)

def main():
    """Main function to extract and summarize emissions targets from pre-computed search results"""
    # Load pre-computed search results
    combined_df = load_search_results()
    
    if combined_df is None or combined_df.empty:
        logger.warning("No search results to process.")
        return
    
    # Load validation data
    validation_df = load_validation_data()
    
    # Extract and summarize targets
    summary_df = summarize_targets(combined_df, validation_df)
    logger.info(f"Generated summary for {len(summary_df)} countries")
    
    # Format and print summary
    formatted_summary = format_summary(summary_df)
    print(formatted_summary)
    
    # Create results directory
    results_dir = os.path.join('data', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save final result to CSV
    final_path = os.path.join(results_dir, 'final_result.csv')
    summary_df.to_csv(final_path, index=False)
    logger.info(f"Final result saved to {final_path}")

if __name__ == "__main__":
    main()
