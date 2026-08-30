# Labeling Service for Fine-tuning Dataset Annotation
import sqlite3
from pathlib import Path
from datetime import datetime
import json
import base64
import numpy as np
from typing import List, Dict, Optional

class LabelingService:
    """
    Manages image labeling for fine-tuning dataset creation.
    Stores labels, tracks annotation progress, exports for training.
    """

    def __init__(self, db_path: str = "labeling.db"):
        """Initialize labeling database"""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create labeling database schema"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Images table: stores image metadata
        c.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT UNIQUE,
                image_bytes BLOB,
                source TEXT,              -- 'webcam', 'upload', 'mobile'
                uploaded_at TEXT,
                width INTEGER,
                height INTEGER,
                status TEXT DEFAULT 'pending'  -- 'pending', 'labeled', 'training'
            )
        """)

        # Labels table: stores annotations
        c.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                class_name TEXT,          -- e.g., 'person', 'car', 'dog'
                confidence REAL,          -- 1.0 for manual, auto-detected
                labeled_by TEXT,          -- username or 'auto'
                labeled_at TEXT,
                bbox_x1 REAL,             -- Optional: bounding box
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                notes TEXT,
                FOREIGN KEY(image_id) REFERENCES images(id)
            )
        """)

        # Classes table: manage class categories
        c.execute("""
            CREATE TABLE IF NOT EXISTS classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT UNIQUE,
                color TEXT,               -- hex color for UI
                description TEXT,
                created_at TEXT
            )
        """)

        # Labeling sessions table: track annotation progress
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT,
                created_at TEXT,
                completed_at TEXT,
                labeled_count INTEGER DEFAULT 0,
                total_count INTEGER,
                status TEXT DEFAULT 'active'  -- 'active', 'completed'
            )
        """)

        conn.commit()
        conn.close()

    def add_image(self, image_bytes: bytes, source: str, width: int, height: int) -> int:
        """Add image to labeling queue"""
        # Create hash for deduplication
        image_hash = hash(image_bytes) % (10 ** 10)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        try:
            c.execute("""
                INSERT INTO images (image_hash, image_bytes, source, uploaded_at, width, height)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (image_hash, image_bytes, source, datetime.now().isoformat(), width, height))

            image_id = c.lastrowid
            conn.commit()
            return image_id
        except sqlite3.IntegrityError:
            # Image already exists
            c.execute("SELECT id FROM images WHERE image_hash = ?", (image_hash,))
            return c.fetchone()[0]
        finally:
            conn.close()

    def label_image(self, image_id: int, class_name: str, labeled_by: str = "user",
                   bbox: Optional[Dict] = None, notes: str = "") -> int:
        """Label an image with class and optional bounding box"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Register class if new
        self._ensure_class_exists(class_name)

        # Insert label
        bbox_x1 = bbox.get('x1') if bbox else None
        bbox_y1 = bbox.get('y1') if bbox else None
        bbox_x2 = bbox.get('x2') if bbox else None
        bbox_y2 = bbox.get('y2') if bbox else None

        c.execute("""
            INSERT INTO labels (image_id, class_name, confidence, labeled_by, labeled_at,
                              bbox_x1, bbox_y1, bbox_x2, bbox_y2, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (image_id, class_name, 1.0, labeled_by, datetime.now().isoformat(),
              bbox_x1, bbox_y1, bbox_x2, bbox_y2, notes))

        label_id = c.lastrowid

        # Update image status
        c.execute("UPDATE images SET status = ? WHERE id = ?", ('labeled', image_id))

        conn.commit()
        conn.close()

        return label_id

    def _ensure_class_exists(self, class_name: str):
        """Create class if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT id FROM classes WHERE class_name = ?", (class_name,))
        if c.fetchone() is None:
            # Assign random color for new class
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
            color = colors[len(class_name) % len(colors)]

            c.execute("""
                INSERT INTO classes (class_name, color, created_at)
                VALUES (?, ?, ?)
            """, (class_name, color, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def get_unlabeled_images(self, limit: int = 10) -> List[Dict]:
        """Get pending images for labeling"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT id, image_bytes, source, uploaded_at, width, height
            FROM images
            WHERE status = 'pending'
            LIMIT ?
        """, (limit,))

        images = []
        for row in c.fetchall():
            # Encode image as base64 for UI
            img_b64 = base64.b64encode(row['image_bytes']).decode('utf-8')
            images.append({
                'id': row['id'],
                'image': img_b64,
                'source': row['source'],
                'uploaded_at': row['uploaded_at'],
                'width': row['width'],
                'height': row['height']
            })

        conn.close()
        return images

    def get_labeled_images(self, session_id: Optional[int] = None, limit: int = 20) -> List[Dict]:
        """Get labeled images for review/export"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT i.id, i.image_bytes, i.source, i.uploaded_at,
                   l.class_name, l.bbox_x1, l.bbox_y1, l.bbox_x2, l.bbox_y2, l.notes
            FROM images i
            LEFT JOIN labels l ON i.id = l.image_id
            WHERE i.status = 'labeled'
            ORDER BY i.uploaded_at DESC
            LIMIT ?
        """, (limit,))

        images = []
        for row in c.fetchall():
            img_b64 = base64.b64encode(row['image_bytes']).decode('utf-8')
            images.append({
                'id': row['id'],
                'image': img_b64,
                'source': row['source'],
                'class': row['class_name'],
                'bbox': {
                    'x1': row['bbox_x1'],
                    'y1': row['bbox_y1'],
                    'x2': row['bbox_x2'],
                    'y2': row['bbox_y2']
                } if row['bbox_x1'] else None,
                'notes': row['notes']
            })

        conn.close()
        return images

    def get_stats(self) -> Dict:
        """Get labeling statistics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM images WHERE status = 'pending'")
        pending = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM images WHERE status = 'labeled'")
        labeled = c.fetchone()[0]

        c.execute("SELECT COUNT(DISTINCT class_name) FROM labels")
        num_classes = c.fetchone()[0]

        c.execute("SELECT class_name, COUNT(*) as count FROM labels GROUP BY class_name")
        class_distribution = {row[0]: row[1] for row in c.fetchall()}

        conn.close()

        return {
            'pending': pending,
            'labeled': labeled,
            'total': pending + labeled,
            'num_classes': num_classes,
            'class_distribution': class_distribution,
            'progress': (labeled / (pending + labeled) * 100) if (pending + labeled) > 0 else 0
        }

    def export_for_training(self, output_path: str = "training_dataset.json") -> Dict:
        """Export labeled data for fine-tuning"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT i.id, i.image_bytes, l.class_name
            FROM images i
            JOIN labels l ON i.id = l.image_id
            WHERE i.status = 'labeled'
        """)

        dataset = {
            'images': [],
            'labels': [],
            'classes': [],
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'total_samples': 0
            }
        }

        class_to_idx = {}
        for i, row in enumerate(c.fetchall()):
            class_name = row['class_name']

            # Map class name to index
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)
                dataset['classes'].append(class_name)

            # Add image and label
            img_b64 = base64.b64encode(row['image_bytes']).decode('utf-8')
            dataset['images'].append(img_b64)
            dataset['labels'].append(class_to_idx[class_name])

        dataset['metadata']['total_samples'] = len(dataset['labels'])

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(dataset, f)

        conn.close()

        return {
            'status': 'success',
            'total_samples': dataset['metadata']['total_samples'],
            'classes': dataset['classes'],
            'saved_to': output_path
        }
