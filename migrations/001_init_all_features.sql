-- ============================================================================
-- Sentinel Vision - Database Migrations for All 20 Features
-- Run: psql -U user -d sentinel_db -f migrations/001_init_all_features.sql
-- ============================================================================

-- ============================================================================
-- CORE TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS teams (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    owner_id INTEGER NOT NULL REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS team_members (
    id SERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    role VARCHAR(50) DEFAULT 'member',
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, user_id)
);

-- ============================================================================
-- 1. IMAGE ASSETS
-- ============================================================================

CREATE TABLE IF NOT EXISTS image_assets (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(512) NOT NULL,
    s3_key VARCHAR(512),
    owner_id INTEGER NOT NULL REFERENCES users(id),
    size_mb FLOAT,
    format VARCHAR(50),
    width INTEGER,
    height INTEGER,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    tags VARCHAR(500),
    INDEX user_idx (owner_id),
    INDEX created_idx (uploaded_at)
);

-- ============================================================================
-- 2. ML MODELS & VERSIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    model_type VARCHAR(50),
    framework VARCHAR(50),
    version VARCHAR(50),
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    inference_time_ms FLOAT,
    model_size_mb FLOAT,
    is_active BOOLEAN DEFAULT TRUE,
    is_production BOOLEAN DEFAULT FALSE,
    file_path VARCHAR(512),
    s3_key VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    INDEX type_idx (model_type),
    INDEX active_idx (is_active)
);

CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES ml_models(id),
    version VARCHAR(50) NOT NULL,
    file_path VARCHAR(512),
    s3_key VARCHAR(512),
    accuracy FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployed_at TIMESTAMP,
    is_production BOOLEAN DEFAULT FALSE,
    changelog TEXT,
    UNIQUE(model_id, version)
);

-- ============================================================================
-- 3. DATASETS
-- ============================================================================

CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id INTEGER NOT NULL REFERENCES users(id),
    total_images INTEGER DEFAULT 0,
    total_annotations INTEGER DEFAULT 0,
    size_mb FLOAT DEFAULT 0.0,
    status VARCHAR(50) DEFAULT 'collecting',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    split_train FLOAT DEFAULT 0.8,
    split_val FLOAT DEFAULT 0.1,
    split_test FLOAT DEFAULT 0.1,
    metadata JSONB,
    INDEX owner_idx (owner_id),
    INDEX status_idx (status)
);

-- ============================================================================
-- 4. TRAINING JOBS
-- ============================================================================

CREATE TABLE IF NOT EXISTS training_jobs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    dataset_id INTEGER REFERENCES datasets(id),
    model_type VARCHAR(50),
    job_name VARCHAR(255),
    status VARCHAR(50) DEFAULT 'queued',
    epochs INTEGER,
    batch_size INTEGER,
    learning_rate FLOAT,
    optimizer VARCHAR(50) DEFAULT 'adam',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    accuracy FLOAT,
    loss FLOAT,
    val_accuracy FLOAT,
    val_loss FLOAT,
    output_model_id INTEGER REFERENCES ml_models(id),
    logs TEXT,
    error_message TEXT,
    INDEX status_idx (status),
    INDEX user_idx (user_id)
);

-- ============================================================================
-- 5. BATCH JOBS
-- ============================================================================

CREATE TABLE IF NOT EXISTS batch_jobs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    model_id INTEGER REFERENCES ml_models(id),
    dataset_id INTEGER REFERENCES datasets(id),
    status VARCHAR(50) DEFAULT 'pending',
    total_items INTEGER,
    processed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    results_location VARCHAR(512),
    metadata JSONB,
    INDEX status_idx (status)
);

-- ============================================================================
-- 6. ANNOTATIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS annotations (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES image_assets(id),
    annotator_id INTEGER NOT NULL REFERENCES users(id),
    annotation_type VARCHAR(50),
    label VARCHAR(255),
    confidence FLOAT,
    data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    verified BOOLEAN DEFAULT FALSE,
    verified_by INTEGER REFERENCES users(id),
    verified_at TIMESTAMP,
    INDEX image_idx (image_id)
);

-- ============================================================================
-- 7. INFERENCE RESULTS & METRICS
-- ============================================================================

CREATE TABLE IF NOT EXISTS inference_results (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id),
    image_id INTEGER REFERENCES image_assets(id),
    user_id INTEGER REFERENCES users(id),
    predictions JSONB,
    confidence_score FLOAT,
    inference_time_ms FLOAT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX model_idx (model_id),
    INDEX time_idx (processed_at)
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id),
    metric_name VARCHAR(100),
    value FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dataset_id INTEGER,
    metadata JSONB,
    INDEX model_time_idx (model_id, timestamp)
);

-- ============================================================================
-- 8. A/B TESTING
-- ============================================================================

CREATE TABLE IF NOT EXISTS ab_experiments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    description TEXT,
    model_a_id INTEGER REFERENCES ml_models(id),
    model_b_id INTEGER REFERENCES ml_models(id),
    status VARCHAR(50) DEFAULT 'running',
    sample_size INTEGER,
    confidence_level FLOAT DEFAULT 0.95,
    total_predictions INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    winner_model_id INTEGER,
    p_value FLOAT,
    results JSONB
);

-- ============================================================================
-- 9. NOTIFICATIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS notifications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    type VARCHAR(50),
    title VARCHAR(255),
    message TEXT,
    read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data JSONB,
    INDEX user_idx (user_id),
    INDEX read_idx (read)
);

-- ============================================================================
-- 10. API KEYS
-- ============================================================================

CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    key_prefix VARCHAR(10),
    name VARCHAR(255),
    permissions JSONB,
    rate_limit INTEGER DEFAULT 100,
    last_used TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX key_hash_idx (key_hash)
);

-- ============================================================================
-- 11. WEBHOOKS
-- ============================================================================

CREATE TABLE IF NOT EXISTS webhooks (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    url VARCHAR(512),
    events JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_triggered TIMESTAMP,
    failure_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id SERIAL PRIMARY KEY,
    webhook_id INTEGER REFERENCES webhooks(id),
    event VARCHAR(100),
    payload JSONB,
    response_status INTEGER,
    response_body TEXT,
    delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    retry_count INTEGER DEFAULT 0
);

-- ============================================================================
-- 12. SCHEDULED JOBS
-- ============================================================================

CREATE TABLE IF NOT EXISTS scheduled_jobs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    name VARCHAR(255),
    job_type VARCHAR(50),
    cron_expression VARCHAR(100),
    config JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    last_run TIMESTAMP,
    next_run TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_result VARCHAR(50),
    last_error TEXT
);

-- ============================================================================
-- 13. COST TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS cost_metrics (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    operation VARCHAR(50),
    cost FLOAT DEFAULT 0.0,
    units INTEGER,
    unit_cost FLOAT,
    date DATE DEFAULT CURRENT_DATE,
    metadata JSONB,
    INDEX user_date_idx (user_id, date)
);

-- ============================================================================
-- 14. ANOMALY DETECTION
-- ============================================================================

CREATE TABLE IF NOT EXISTS anomaly_detections (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    image_id INTEGER REFERENCES image_assets(id),
    model_id INTEGER REFERENCES ml_models(id),
    anomaly_score FLOAT,
    is_anomaly BOOLEAN,
    heatmap_path VARCHAR(512),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    INDEX time_idx (detected_at)
);

-- ============================================================================
-- 15. EXPLAINABILITY
-- ============================================================================

CREATE TABLE IF NOT EXISTS explainability_reports (
    id SERIAL PRIMARY KEY,
    inference_id INTEGER REFERENCES inference_results(id),
    method VARCHAR(50),
    explanation_data JSONB,
    heatmap_path VARCHAR(512),
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- ============================================================================
-- 16. AUDIT LOGS
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    action VARCHAR(100),
    resource VARCHAR(255),
    resource_id INTEGER,
    status_code INTEGER,
    ip_address VARCHAR(50),
    user_agent VARCHAR(500),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    INDEX user_idx (user_id),
    INDEX action_idx (action),
    INDEX time_idx (timestamp)
);

-- ============================================================================
-- 17. EXPORT JOBS
-- ============================================================================

CREATE TABLE IF NOT EXISTS export_jobs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    source_type VARCHAR(50),
    source_id INTEGER,
    format VARCHAR(50),
    status VARCHAR(50) DEFAULT 'pending',
    output_path VARCHAR(512),
    s3_key VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    file_size_mb FLOAT,
    INDEX status_idx (status)
);

-- ============================================================================
-- 18. STREAMING SESSIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS streaming_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    model_id INTEGER REFERENCES ml_models(id),
    source VARCHAR(255),
    status VARCHAR(50) DEFAULT 'inactive',
    frames_processed INTEGER DEFAULT 0,
    detections INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    stopped_at TIMESTAMP,
    avg_fps FLOAT,
    metadata JSONB
);

-- ============================================================================
-- 19. CUSTOM MODEL TRAINING
-- ============================================================================

CREATE TABLE IF NOT EXISTS custom_model_training (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    name VARCHAR(255),
    description TEXT,
    base_model_id INTEGER REFERENCES ml_models(id),
    dataset_id INTEGER REFERENCES datasets(id),
    architecture VARCHAR(255),
    hyperparameters JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    accuracy FLOAT,
    loss FLOAT,
    output_model_id INTEGER REFERENCES ml_models(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    INDEX status_idx (status)
);

-- ============================================================================
-- 20. INVOICES & BILLING
-- ============================================================================

CREATE TABLE IF NOT EXISTS invoices (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    invoice_number VARCHAR(50) UNIQUE NOT NULL,
    period_start DATE,
    period_end DATE,
    total_cost FLOAT,
    subtotal FLOAT,
    tax FLOAT,
    status VARCHAR(50) DEFAULT 'draft',
    due_date DATE,
    paid_date DATE,
    issued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    items JSONB,
    metadata JSONB,
    INDEX number_idx (invoice_number),
    INDEX user_idx (user_id)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

CREATE INDEX IF NOT EXISTS users_email_idx ON users(email);
CREATE INDEX IF NOT EXISTS image_assets_owner_idx ON image_assets(owner_id);
CREATE INDEX IF NOT EXISTS training_jobs_user_idx ON training_jobs(user_id);
CREATE INDEX IF NOT EXISTS batch_jobs_user_idx ON batch_jobs(user_id);
CREATE INDEX IF NOT EXISTS annotations_image_idx ON annotations(image_id);
CREATE INDEX IF NOT EXISTS notifications_user_read_idx ON notifications(user_id, read);
CREATE INDEX IF NOT EXISTS audit_logs_user_idx ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS cost_metrics_user_date_idx ON cost_metrics(user_id, date);

-- ============================================================================
-- DONE
-- ============================================================================

-- Verify all tables created
SELECT 'Database initialized successfully' as status;
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;
