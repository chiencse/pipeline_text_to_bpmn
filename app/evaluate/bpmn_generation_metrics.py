

import json
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("ERROR: psycopg2 not installed. Install with: pip install psycopg2-binary")
    exit(1)

from app.logs.db_logger import get_connection, return_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SIMILARITY_SCORE = 0.93

def get_bpmn_feedback_data() -> List[Dict[str, Any]]:
    """
    Get all feedback records for BPMN generation stage (is_mapping = false).
    Ordered by thread_id and timestamp.
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                id, timestamp, thread_id, node_ids, user_decision,
                is_mapping, node_mapping_feedback, bpmn, mapping, user_feedback
            FROM feedback_log
            WHERE is_mapping = false
            ORDER BY thread_id, timestamp
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        logger.info(f"Retrieved {len(results)} BPMN feedback records")
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error(f"Failed to fetch feedback data: {e}")
        raise
    finally:
        if conn:
            return_connection(conn)


def get_mapping_feedback_data() -> List[Dict[str, Any]]:
    """
    Get all feedback records for mapping stage (is_mapping = true).
    Ordered by thread_id and timestamp.
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                id, timestamp, thread_id, node_ids, user_decision,
                is_mapping, node_mapping_feedback, bpmn, mapping, user_feedback
            FROM feedback_log
            WHERE is_mapping = true
            ORDER BY thread_id, timestamp
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        logger.info(f"Retrieved {len(results)} mapping feedback records")
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error(f"Failed to fetch mapping feedback data: {e}")
        raise
    finally:
        if conn:
            return_connection(conn)

def get_retrieval_scores_by_thread() -> List[Dict[str, Any]]:
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT DISTINCT 
                thread_id,  node_id, avg_total_score
            FROM retrieval_scores
            ORDER BY thread_id;
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        logger.info(f"Retrieved {len(results)} mapping feedback records")
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error(f"Failed to fetch mapping feedback data: {e}")
        raise
    finally:
        if conn:
            return_connection(conn)


def extract_node_ids_from_bpmn(bpmn: Dict[str, Any]) -> set:
    """Extract all node IDs from BPMN JSON structure."""
    if not bpmn or 'nodes' not in bpmn:
        return set()
    return {node['id'] for node in bpmn['nodes']}


def extract_nodes_with_names_from_bpmn(bpmn: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract nodes with both ID and name from BPMN JSON structure.
    Returns dict: {node_id: node_name}
    """
    if not bpmn or 'nodes' not in bpmn:
        return {}
    return {node['id']: node.get('name', '') for node in bpmn['nodes']}


def group_feedback_by_thread(feedback_records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group feedback records by thread_id."""
    grouped = defaultdict(list)
    for record in feedback_records:
        grouped[record['thread_id']].append(record)
    return dict(grouped)


def calculate_metrics_for_thread(thread_feedbacks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate TP, FP, FN for a single thread (to avoid double counting).
    
    Strategy similar to mapping:
    1. Collect ALL node_ids from all feedbacks (union)
    2. Get first and final BPMN states
    3. Calculate metrics based on final state:
       - FP: all_node_ids (nodes user had to feedback on - problematic)
       - TP: nodes in final BPMN NOT in all_node_ids (correctly kept)
       - FN: nodes in final BPMN NOT in first BPMN (missing, had to add)
    
    Args:
        thread_feedbacks: List of feedback records for a thread, ordered by timestamp
    
    Returns:
        Dictionary with aggregated metrics and metadata
    """
    if not thread_feedbacks:
        return None
    
    first_feedback = thread_feedbacks[0]
    last_feedback = thread_feedbacks[-1]
    thread_id = first_feedback['thread_id']
    
    # Check if approved on first try
    first_approved = (len(thread_feedbacks) == 1 and 
                     first_feedback['user_decision'] == 'approve')
    
    # Check if finally approved
    final_approved = last_feedback['user_decision'] == 'approve'
    
    # Collect ALL node_ids from all feedbacks (union)
    all_node_ids = set()
    for feedback in thread_feedbacks:
        node_ids = feedback.get('node_ids', [])
        if node_ids:
            all_node_ids.update(node_ids)
    
    # Get first and final BPMN node sets
    first_nodes_dict = extract_nodes_with_names_from_bpmn(first_feedback.get('bpmn'))
    final_nodes_dict = extract_nodes_with_names_from_bpmn(last_feedback.get('bpmn'))
    
    first_nodes_set = set(first_nodes_dict.keys())
    final_nodes_set = set(final_nodes_dict.keys())
    
    # Special case: If approved on first try, all nodes are TP
    if first_approved:
        tp_nodes = list(first_nodes_set)
        fp_nodes = []
        fn_nodes = []
    else:
        # FP: all_node_ids (nodes user had to give feedback on = problematic)
        fp_nodes = list(all_node_ids)
        
        # TP: nodes in BOTH first AND final BPMN, NOT in all_node_ids
        # (correctly kept from start to end without feedback)
        # Must also have same name (unchanged)
        tp_nodes_set = set()
        for node_id in first_nodes_set & final_nodes_set:
            # Check if node is unchanged (same name) and not in all_node_ids
            if (first_nodes_dict[node_id] == final_nodes_dict[node_id] and 
                node_id not in all_node_ids):
                tp_nodes_set.add(node_id)
        tp_nodes = list(tp_nodes_set)
        
        # FN: nodes in final BPMN but NOT in first BPMN (missing, had to add)
        # Also check for nodes with same ID but different name (renamed/modified)
        fn_nodes_set = set()
        
        # 1. Completely new nodes
        new_nodes = final_nodes_set - first_nodes_set
        fn_nodes_set.update(new_nodes)
        
        # 2. Nodes with same ID but different name (renamed/modified)
        for node_id in first_nodes_set & final_nodes_set:
            if first_nodes_dict[node_id] != final_nodes_dict[node_id]:
                fn_nodes_set.add(node_id)
        
        fn_nodes = list(fn_nodes_set)
    
    return {
        'thread_id': thread_id,
        'num_feedbacks': len(thread_feedbacks),
        'first_approved': first_approved,
        'final_approved': final_approved,
        'first_nodes_count': len(first_nodes_set),
        'final_nodes_count': len(final_nodes_set),
        'tp_count': len(tp_nodes),
        'fp_count': len(fp_nodes),
        'fn_count': len(fn_nodes),
        'tp_nodes': tp_nodes,
        'fp_nodes': fp_nodes,
        'fn_nodes': fn_nodes,
        'all_node_ids': list(all_node_ids)
    }


def calculate_overall_metrics(thread_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall metrics across all threads."""
    total_threads = len(thread_metrics)
    
    if total_threads == 0:
        return {
            'total_threads': 0,
            'first_approval_count': 0,
            'first_approval_rate': 0.0,
            'final_approval_count': 0,
            'final_approval_rate': 0.0,
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
            'avg_tp_per_thread': 0.0,
            'avg_fp_per_thread': 0.0,
            'avg_fn_per_thread': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    # Count approvals
    first_approval_count = sum(1 for m in thread_metrics if m['first_approved'])
    final_approval_count = sum(1 for m in thread_metrics if m['final_approved'])
    
    # Sum TP, FP, FN
    total_tp = sum(m['tp_count'] for m in thread_metrics)
    total_fp = sum(m['fp_count'] for m in thread_metrics)
    total_fn = sum(m['fn_count'] for m in thread_metrics)
    
    # Calculate rates and averages
    first_approval_rate = first_approval_count / total_threads
    final_approval_rate = final_approval_count / total_threads
    
    avg_tp = total_tp / total_threads
    avg_fp = total_fp / total_threads
    avg_fn = total_fn / total_threads
    
    # Calculate Precision, Recall, F1
    # Precision = TP / (TP + FP)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    
    # Recall = TP / (TP + FN)
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'total_threads': total_threads,
        'first_approval_count': first_approval_count,
        'first_approval_rate': first_approval_rate,
        'final_approval_count': final_approval_count,
        'final_approval_rate': final_approval_rate,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'avg_tp_per_thread': avg_tp,
        'avg_fp_per_thread': avg_fp,
        'avg_fn_per_thread': avg_fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def print_results(overall_metrics: Dict[str, Any], thread_metrics: List[Dict[str, Any]]):
    """Print formatted results to console only (no text file output)."""
    print("\n" + "="*80)
    print("BPMN NODE GENERATION CORRECTNESS METRICS")
    print("="*80)
    
    print("\n--- OVERALL METRICS ---")
    print(f"Total threads analyzed: {overall_metrics['total_threads']}")
    print(f"\nApproval Rates:")
    print(f"  First approval count: {overall_metrics['first_approval_count']}")
    print(f"  First approval rate: {overall_metrics['first_approval_rate']:.2%}")
    print(f"  Final approval count: {overall_metrics['final_approval_count']}")
    print(f"  Final approval rate: {overall_metrics['final_approval_rate']:.2%}")
    
    print(f"\nNode Generation Metrics (TP/FP/FN):")
    print(f"  Total TP (True Positives): {overall_metrics['total_tp']}")
    print(f"  Total FP (False Positives): {overall_metrics['total_fp']}")
    print(f"  Total FN (False Negatives): {overall_metrics['total_fn']}")
    print(f"  Average TP per thread: {overall_metrics['avg_tp_per_thread']:.2f}")
    print(f"  Average FP per thread: {overall_metrics['avg_fp_per_thread']:.2f}")
    print(f"  Average FN per thread: {overall_metrics['avg_fn_per_thread']:.2f}")
    
    print(f"\nQuality Metrics:")
    print(f"  Precision (TP/(TP+FP)): {overall_metrics['precision']:.2%}")
    print(f"  Recall (TP/(TP+FN)): {overall_metrics['recall']:.2%}")
    print(f"  F1 Score: {overall_metrics['f1_score']:.2%}")
    
    print("\n" + "-"*80)
    print("PER-THREAD SUMMARY")
    print("-"*80)
    
    for i, metrics in enumerate(thread_metrics, 1):
        print(f"\n[Thread {i}] {metrics['thread_id']}")
        print(f"  Feedbacks: {metrics['num_feedbacks']}")
        print(f"  First approved: {metrics['first_approved']}")
        print(f"  Final approved: {metrics['final_approved']}")
        print(f"  Nodes (first → final): {metrics['first_nodes_count']} → {metrics['final_nodes_count']}")
        print(f"  All node_ids (union): {len(metrics.get('all_node_ids', []))} nodes")
        print(f"  TP: {metrics['tp_count']} | FP: {metrics['fp_count']} | FN: {metrics['fn_count']}")
        
        if metrics.get('fp_nodes'):
            print(f"    FP nodes (problematic): {metrics['fp_nodes']}")
        if metrics.get('fn_nodes'):
            print(f"    FN nodes (missing/modified): {metrics['fn_nodes']}")
    
    print("\n" + "="*80)


def calculate_mapping_metrics_for_thread(thread_feedbacks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate TP, FP, FN for all mapping feedbacks in a thread.
    
    Strategy to avoid double counting:
    1. Collect ALL node_ids from all feedbacks (union)
    2. Get final mapping state (from last feedback or approved feedback)
    3. Classify nodes based on final mapping:
       - FP: node_ids that have is_automatic=True (wrongly predicted as automatic)
       - FN: node_ids that have is_automatic=False (should be automatic)
       - TP: nodes with is_automatic=True NOT in node_ids (correctly automated)
    
    Args:
        thread_feedbacks: List of mapping feedback records for a thread, ordered by timestamp
    
    Returns:
        Dictionary with TP/FP/FN counts and node lists
    """
    if not thread_feedbacks:
        return None
    
    first_feedback = thread_feedbacks[0]
    last_feedback = thread_feedbacks[-1]
    thread_id = first_feedback['thread_id']
    
    # Check if approved
    first_approved = (len(thread_feedbacks) == 1 and 
                     first_feedback['user_decision'] == 'approve')
    final_approved = last_feedback['user_decision'] == 'approve'
    
    # Collect ALL node_ids from all feedbacks (union)
    all_node_ids = set()
    for feedback in thread_feedbacks:
        node_ids = feedback.get('node_ids', [])
        if node_ids:
            all_node_ids.update(node_ids)
    
    # Get final mapping state (use last feedback's mapping)
    final_mapping = last_feedback.get('mapping', [])
    
    if not final_mapping:
        return {
            'thread_id': thread_id,
            'num_feedbacks': len(thread_feedbacks),
            'first_approved': first_approved,
            'final_approved': final_approved,
            'tp_count': 0,
            'fp_count': 0,
            'fn_count': 0,
            'tp_nodes': [],
            'fp_nodes': [],
            'fn_nodes': [],
            'total_nodes': 0,
            'all_node_ids': list(all_node_ids)
        }
    
    # Parse final mapping to get nodes with is_automatic status
    nodes_automatic_true = set()
    nodes_automatic_false = set()
    
    for node_item in final_mapping:
        if not isinstance(node_item, dict):
            continue
        
        for node_id, node_info in node_item.items():
            if not isinstance(node_info, dict):
                continue
            
            is_automatic = node_info.get('is_automatic', False)
            if is_automatic:
                nodes_automatic_true.add(node_id)
            else:
                nodes_automatic_false.add(node_id)
    
    # Calculate TP, FP, FN based on all_node_ids and final mapping
    # FP: node_ids in automatic_true (wrongly predicted as automatic, but user had to feedback)
    fp_nodes = list(all_node_ids & nodes_automatic_true)
    
    # FN: node_ids in automatic_false (should be automatic, but marked as manual)
    fn_nodes = list(all_node_ids & nodes_automatic_false)
    
    # TP: nodes in automatic_true NOT in all_node_ids (correctly predicted as automatic)
    tp_nodes = list(nodes_automatic_true - all_node_ids)
    
    return {
        'thread_id': thread_id,
        'num_feedbacks': len(thread_feedbacks),
        'first_approved': first_approved,
        'final_approved': final_approved,
        'tp_count': len(tp_nodes),
        'fp_count': len(fp_nodes),
        'fn_count': len(fn_nodes),
        'tp_nodes': tp_nodes,
        'fp_nodes': fp_nodes,
        'fn_nodes': fn_nodes,
        'total_nodes': len(final_mapping),
        'total_automatic_true': len(nodes_automatic_true),
        'total_automatic_false': len(nodes_automatic_false),
        'all_node_ids': list(all_node_ids)
    }


def calculate_overall_mapping_metrics(thread_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall mapping metrics across all threads."""
    total_threads = len(thread_metrics)
    
    if total_threads == 0:
        return {
            'total_threads': 0,
            'first_approval_count': 0,
            'first_approval_rate': 0.0,
            'final_approval_count': 0,
            'final_approval_rate': 0.0,
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
            'avg_tp_per_thread': 0.0,
            'avg_fp_per_thread': 0.0,
            'avg_fn_per_thread': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    # Count approvals
    first_approval_count = sum(1 for m in thread_metrics if m['first_approved'])
    final_approval_count = sum(1 for m in thread_metrics if m['final_approved'])
    
    # Sum TP, FP, FN
    total_tp = sum(m['tp_count'] for m in thread_metrics)
    total_fp = sum(m['fp_count'] for m in thread_metrics)
    total_fn = sum(m['fn_count'] for m in thread_metrics)
    
    # Calculate rates and averages
    first_approval_rate = first_approval_count / total_threads
    final_approval_rate = final_approval_count / total_threads
    avg_tp = total_tp / total_threads
    avg_fp = total_fp / total_threads
    avg_fn = total_fn / total_threads
    
    # Calculate Precision, Recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'total_threads': total_threads,
        'first_approval_count': first_approval_count,
        'first_approval_rate': first_approval_rate,
        'final_approval_count': final_approval_count,
        'final_approval_rate': final_approval_rate,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'avg_tp_per_thread': avg_tp,
        'avg_fp_per_thread': avg_fp,
        'avg_fn_per_thread': avg_fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def print_mapping_results(overall_metrics: Dict[str, Any], thread_metrics: List[Dict[str, Any]]):
    """Print formatted mapping results to console."""
    print("\n" + "="*80)
    print("MAPPING AUTOMATION CORRECTNESS METRICS")
    print("="*80)
    
    print("\n--- OVERALL METRICS ---")
    print(f"Total threads analyzed: {overall_metrics['total_threads']}")
    print(f"\nApproval Rates:")
    print(f"  First approval count: {overall_metrics['first_approval_count']}")
    print(f"  First approval rate: {overall_metrics['first_approval_rate']:.2%}")
    print(f"  Final approval count: {overall_metrics['final_approval_count']}")
    print(f"  Final approval rate: {overall_metrics['final_approval_rate']:.2%}")
    
    print(f"\nAutomation Prediction Metrics (TP/FP/FN):")
    print(f"  Total TP (Correctly Automated): {overall_metrics['total_tp']}")
    print(f"  Total FP (Wrongly Automated): {overall_metrics['total_fp']}")
    print(f"  Total FN (Missed Automation): {overall_metrics['total_fn']}")
    print(f"  Average TP per thread: {overall_metrics['avg_tp_per_thread']:.2f}")
    print(f"  Average FP per thread: {overall_metrics['avg_fp_per_thread']:.2f}")
    print(f"  Average FN per thread: {overall_metrics['avg_fn_per_thread']:.2f}")
    
    print(f"\nQuality Metrics:")
    print(f"  Precision (TP/(TP+FP)): {overall_metrics['precision']:.2%}")
    print(f"  Recall (TP/(TP+FN)): {overall_metrics['recall']:.2%}")
    print(f"  F1 Score: {overall_metrics['f1_score']:.2%}")
    
    print("\n" + "-"*80)
    print("PER-THREAD SUMMARY")
    print("-"*80)
    
    for i, metrics in enumerate(thread_metrics, 1):
        print(f"\n[Thread {i}] {metrics['thread_id']}")
        print(f"  Feedbacks: {metrics['num_feedbacks']}")
        print(f"  First approved: {metrics['first_approved']}")
        print(f"  Final approved: {metrics['final_approved']}")
        print(f"  Total nodes in mapping: {metrics['total_nodes']}")
        print(f"  Nodes with is_automatic=True: {metrics.get('total_automatic_true', 0)}")
        print(f"  Nodes with is_automatic=False: {metrics.get('total_automatic_false', 0)}")
        print(f"  All node_ids (union): {len(metrics.get('all_node_ids', []))} nodes")
        print(f"  TP: {metrics['tp_count']} | FP: {metrics['fp_count']} | FN: {metrics['fn_count']}")
        
        if metrics['fp_nodes']:
            print(f"    FP nodes (wrongly automated): {metrics['fp_nodes']}")
        if metrics['fn_nodes']:
            print(f"    FN nodes (missed automation): {metrics['fn_nodes']}")
    
    print("\n" + "="*80)


def save_json_results(bpmn_overall: Dict[str, Any], bpmn_threads: List[Dict[str, Any]], 
                     mapping_overall: Dict[str, Any] = None, mapping_threads: List[Dict[str, Any]] = None, similarity_score_mapping: Dict[str, Any] = None  ):
    """Save results to JSON file for further analysis."""
    try:
        from datetime import datetime
        
        # Prepare BPMN data for JSON export
        bpmn_threads_json = []
        for m in bpmn_threads:
            thread_data = {
                'thread_id': m['thread_id'],
                'num_feedbacks': m['num_feedbacks'],
                'first_approved': m['first_approved'],
                'final_approved': m['final_approved'],
                'first_nodes_count': m['first_nodes_count'],
                'final_nodes_count': m['final_nodes_count'],
                'tp_count': m['tp_count'],
                'fp_count': m['fp_count'],
                'fn_count': m['fn_count'],
                'tp_nodes': m['tp_nodes'],
                'fp_nodes': m['fp_nodes'],
                'fn_nodes': m['fn_nodes'],
                'all_node_ids': m.get('all_node_ids', [])
            }
            
            bpmn_threads_json.append(thread_data)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'bpmn_generation': {
                'overall_metrics': bpmn_overall,
                'thread_metrics': bpmn_threads_json
            }
        }
        
        # Add mapping data if available
        if mapping_overall and mapping_threads:
            results['mapping_automation'] = {
                'overall_metrics': mapping_overall,
                'thread_metrics': mapping_threads
            }
        
        if similarity_score_mapping:
            results['similarity_score_mapping'] = similarity_score_mapping
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"app/evaluate/results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON results saved to: {filename}")
        print(f"\nJSON results saved to: {filename}")
        
    except Exception as e:
        logger.warning(f"Failed to save JSON results: {e}")


def get_node_mapping_is_automatic() -> Dict[str, Any]:
    sum_similarity_score = 0
    num_nodes = 0
    num_candidate_in_node_reject = 0
    sum_score_candidate_in_node_reject = 0
    mapping_feedback_records = get_mapping_feedback_data()
    grouped_mapping_feedbacks = group_feedback_by_thread(mapping_feedback_records)
    thread_node_is_automatic = []
    for thread_id, feedbacks in grouped_mapping_feedbacks.items():
        for feedback in feedbacks:
            mapping = feedback.get('mapping', [])
                
            if not mapping:
                continue
            if feedback.get('user_decision', '') == 'reject':
                for node_item in mapping:
                    if not isinstance(node_item, dict):
                        continue
                    for node_id, node_info in node_item.items():

                        if not isinstance(node_info, dict):
                            continue
                        if node_info.get('is_automatic', False) == True:
                            for candidate in node_info.get('candidates', []):
                                if candidate.get('score', 0) > 0:
                                    sum_score_candidate_in_node_reject += candidate.get('score', 0)
                                    num_candidate_in_node_reject += 1
            else:
                for node_item in mapping:
                    if not isinstance(node_item, dict):
                        continue
                    for node_id, node_info in node_item.items():

                        if not isinstance(node_info, dict):
                            continue
                        if node_info.get('is_automatic', False) == True:

                            thread_node_is_automatic.append({
                                'thread_id': thread_id,
                                'node_id': node_id,
                                'is_automatic': node_info.get('is_automatic', False)
                            })
                            for candidate in node_info.get('candidates', []):
                                if candidate.get('score', 0) > 0:
                                    sum_similarity_score += candidate.get('score', 0)
                                    num_nodes += 1
                        
    print("Approval average score: ", num_nodes, (sum_similarity_score/num_nodes)/MAX_SIMILARITY_SCORE)
    print("Reject average score: ", num_candidate_in_node_reject, (sum_score_candidate_in_node_reject/num_candidate_in_node_reject)/MAX_SIMILARITY_SCORE)
    return thread_node_is_automatic, {'approval_average_similiar_candidate_score': (sum_similarity_score / num_nodes)/MAX_SIMILARITY_SCORE, 'reject_average_similiar_candidate_score': (sum_score_candidate_in_node_reject/num_candidate_in_node_reject)/MAX_SIMILARITY_SCORE}

def calculate_similarity_score_mapping() -> Dict[str, Any]:
    thread_node_is_automatic, similarity_score_mapping = get_node_mapping_is_automatic()
  
    retrieval_scores = get_retrieval_scores_by_thread()
    
    sum_similarity_score = 0
    num_nodes = 0
    for record in retrieval_scores:
        thread_id = record['thread_id']
        node_id = record['node_id']
        avg_total_score = record['avg_total_score']
        for element in thread_node_is_automatic:
            if element['thread_id'] == thread_id and element['node_id'] == node_id:
                sum_similarity_score += avg_total_score
                num_nodes += 1
                break
    print(num_nodes, sum_similarity_score/num_nodes, len(thread_node_is_automatic))
    return sum_similarity_score / len(thread_node_is_automatic)



def main():
    """Main execution function."""
    try:
        logger.info("Starting evaluation metrics calculation...")
        
        # ============================================================
        # PART 1: BPMN GENERATION METRICS (is_mapping = false)
        # ============================================================
        logger.info("Part 1: Calculating BPMN generation metrics...")
        
        bpmn_feedback_records = get_bpmn_feedback_data()
        
        if not bpmn_feedback_records:
            print("\n" + "="*80)
            print("No BPMN feedback records found in database.")
            print("="*80)
            bpmn_overall_metrics = None
            bpmn_thread_metrics = []
        else:
            # Group by thread
            grouped_bpmn_feedbacks = group_feedback_by_thread(bpmn_feedback_records)
            logger.info(f"Found {len(grouped_bpmn_feedbacks)} unique threads for BPMN generation")
            
            # Calculate metrics for each thread
            bpmn_thread_metrics = []
            for thread_id, feedbacks in grouped_bpmn_feedbacks.items():
                metrics = calculate_metrics_for_thread(feedbacks)
                if metrics:
                    bpmn_thread_metrics.append(metrics)
            
            # Calculate overall metrics
            bpmn_overall_metrics = calculate_overall_metrics(bpmn_thread_metrics)
            
            # Print results to console
            print_results(bpmn_overall_metrics, bpmn_thread_metrics)
        
        # ============================================================
        # PART 2: MAPPING AUTOMATION METRICS (is_mapping = true)
        # ============================================================
        logger.info("\nPart 2: Calculating mapping automation metrics...")
        
        mapping_feedback_records = get_mapping_feedback_data()
        
        if not mapping_feedback_records:
            print("\n" + "="*80)
            print("No mapping feedback records found in database.")
            print("="*80)
            print("\nPlease ensure:")
            print("  1. Database connection is working")
            print("  2. feedback_log table exists")
            print("  3. There are records with is_mapping = true")
            mapping_overall_metrics = None
            mapping_thread_metrics = []
        else:
            # Group by thread
            grouped_mapping_feedbacks = group_feedback_by_thread(mapping_feedback_records)
            logger.info(f"Found {len(grouped_mapping_feedbacks)} unique threads for mapping automation")
            
            # Calculate metrics for each thread
            mapping_thread_metrics = []
            for thread_id, feedbacks in grouped_mapping_feedbacks.items():
                metrics = calculate_mapping_metrics_for_thread(feedbacks)
                if metrics:
                    mapping_thread_metrics.append(metrics)
            
            # Calculate overall metrics
            mapping_overall_metrics = calculate_overall_mapping_metrics(mapping_thread_metrics)
            
            # Print results to console
            print_mapping_results(mapping_overall_metrics, mapping_thread_metrics)
        
        data,similarity_score_mapping = get_node_mapping_is_automatic()
        save_format_similarity_score_mapping = {
            'approval_average_similiar_candidate_score': similarity_score_mapping.get('approval_average_similiar_candidate_score', 0),
            'reject_average_similiar_candidate_score': similarity_score_mapping.get('reject_average_similiar_candidate_score', 0)
        }
        # ============================================================
        # SAVE COMBINED RESULTS TO JSON
        # ============================================================
        if bpmn_overall_metrics or mapping_overall_metrics:
            save_json_results(
                bpmn_overall=bpmn_overall_metrics or {},
                bpmn_threads=bpmn_thread_metrics,
                mapping_overall=mapping_overall_metrics,
                mapping_threads=mapping_thread_metrics,
                similarity_score_mapping=save_format_similarity_score_mapping

            )
        
        logger.info("Metrics calculation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during metrics calculation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
    # similarity_score_mapping = calculate_similarity_score_mapping()
    # print(similarity_score_mapping)

