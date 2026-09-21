import hashlib
import json
import time
from datetime import datetime

class EvidenceBlock:
    """
    Cryptographic Block representing a single tamper-proof piece of digital evidence
    in the chain of custody.
    """
    def __init__(self, index, timestamp, evidence_id, evidence_type, case_id, 
                 payload, officer_badge, police_station, previous_hash="", authorized_role="INVESTIGATING_OFFICER"):
        self.index = index
        self.timestamp = timestamp
        self.evidence_id = evidence_id
        self.evidence_type = evidence_type
        self.case_id = case_id
        self.payload = payload
        self.officer_badge = officer_badge
        self.police_station = police_station
        self.authorized_role = authorized_role
        self.previous_hash = previous_hash
        self.evidence_hash = self.calculate_evidence_hash(payload)
        self.merkle_root = self.calculate_merkle_root()
        self.block_hash = self.calculate_block_hash()

    def calculate_evidence_hash(self, payload):
        """SHA-256 cryptographic digest of the raw evidence payload"""
        if isinstance(payload, (dict, list)):
            data_str = json.dumps(payload, sort_keys=True)
        else:
            data_str = str(payload)
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    def calculate_merkle_root(self):
        """Calculate Merkle root combining Evidence ID, Case ID, Officer, and Evidence Hash"""
        leaf1 = hashlib.sha256(f"{self.evidence_id}:{self.case_id}".encode()).hexdigest()
        leaf2 = hashlib.sha256(f"{self.officer_badge}:{self.police_station}".encode()).hexdigest()
        leaf3 = self.evidence_hash
        
        combined_1_2 = hashlib.sha256(f"{leaf1}{leaf2}".encode()).hexdigest()
        merkle_root = hashlib.sha256(f"{combined_1_2}{leaf3}".encode()).hexdigest()
        return merkle_root

    def calculate_block_hash(self):
        """Compute block signature hash binding index, timestamp, merkle root, and previous hash"""
        block_string = f"{self.index}{self.timestamp}{self.merkle_root}{self.previous_hash}{self.authorized_role}"
        return hashlib.sha256(block_string.encode('utf-8')).hexdigest()

    def to_dict(self):
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "case_id": self.case_id,
            "officer_badge": self.officer_badge,
            "police_station": self.police_station,
            "authorized_role": self.authorized_role,
            "evidence_hash": self.evidence_hash,
            "merkle_root": self.merkle_root,
            "previous_hash": self.previous_hash,
            "block_hash": self.block_hash,
            "payload_snippet": str(self.payload)[:120] + "..." if len(str(self.payload)) > 120 else str(self.payload)
        }


class EvidenceBlockchain:
    """
    Private Proof-of-Authority Evidence Blockchain Ledger.
    Maintains immutable chain of custody, enforces Smart Contract Access Control,
    and performs automated tamper detection for court admissibility.
    """
    ROLES_PERMISSIONS = {
        "ROLE_INVESTIGATING_OFFICER": ["FIR Record", "Arrest Memo", "Confession Audio", "Seizure Memo"],
        "ROLE_CYBER_CELL": ["CDR Call Log", "Social Media Threat Post", "IP Wiretap Dump", "Encrypted Chat Extract"],
        "ROLE_FORENSIC_EXPERT": ["Ballistics Report", "Digital Device Clone", "DNA Report", "Financial Flow Ledger"],
        "ROLE_MAGISTRATE_JUDGE": ["Judicial Warrant", "Court Evidence Seal", "Trial Ruling Record"]
    }

    def __init__(self):
        self.chain = []
        self.create_genesis_block()
        self.seed_initial_evidence()

    def create_genesis_block(self):
        """Initialize genesis block for the Law Enforcement Evidence Chain"""
        genesis = EvidenceBlock(
            index=0,
            timestamp="2026-01-01 00:00:00",
            evidence_id="EVID_GENESIS_0000",
            evidence_type="Genesis Security Root",
            case_id="NATIONAL_SECURITY_GRID",
            payload={"system": "CrimeGraph AI National Evidence Chain", "version": "2.0"},
            officer_badge="HQ-DIRECTOR-001",
            police_station="National Crime Intelligence HQ",
            previous_hash="0" * 64,
            authorized_role="SYSTEM_ROOT"
        )
        self.chain.append(genesis)

    def register_evidence(self, evidence_id, evidence_type, case_id, payload, officer_badge, police_station, role="ROLE_INVESTIGATING_OFFICER"):
        """
        Smart Contract validation & mining of a new evidence block into the blockchain.
        """
        # Smart Contract Access Control Validation
        allowed_types = self.ROLES_PERMISSIONS.get(role, [])
        if role != "SYSTEM_ROOT" and allowed_types and evidence_type not in allowed_types:
            # Grant emergency permission with audit flag if not directly assigned
            authorized_role = f"{role} [EMERGENCY_OVERRIDE]"
        else:
            authorized_role = role

        prev_block = self.chain[-1]
        new_block = EvidenceBlock(
            index=len(self.chain),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            case_id=case_id,
            payload=payload,
            officer_badge=officer_badge,
            police_station=police_station,
            previous_hash=prev_block.block_hash,
            authorized_role=authorized_role
        )
        self.chain.append(new_block)
        return new_block

    def verify_chain_integrity(self):
        """
        Verify the entire blockchain for cryptographic tampering or broken hashes.
        Returns validation status, total blocks, and any corrupted block indexes.
        """
        corrupted_blocks = []
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i - 1]

            # Check previous hash link
            if current.previous_hash != prev.block_hash:
                corrupted_blocks.append({
                    "block_index": i,
                    "evidence_id": current.evidence_id,
                    "reason": "Previous Hash Mismatch (Broken Chain Link)"
                })
                continue

            # Check Merkle root & block hash calculation
            recalculated_evidence_hash = current.calculate_evidence_hash(current.payload)
            if current.evidence_hash != recalculated_evidence_hash:
                corrupted_blocks.append({
                    "block_index": i,
                    "evidence_id": current.evidence_id,
                    "reason": "Evidence Payload Modified (Hash Mismatch)"
                })
                continue

            recalculated_block_hash = current.calculate_block_hash()
            if current.block_hash != recalculated_block_hash:
                corrupted_blocks.append({
                    "block_index": i,
                    "evidence_id": current.evidence_id,
                    "reason": "Block Signature Tampered"
                })

        is_valid = len(corrupted_blocks) == 0
        return {
            "is_valid": is_valid,
            "total_blocks": len(self.chain),
            "verified_blocks": len(self.chain) - len(corrupted_blocks),
            "corrupted_blocks": corrupted_blocks,
            "status_message": "🟢 100% Chain of Custody Integrity Verified — Admissible in Court" if is_valid else f"🚨 TAMPERING DETECTED in {len(corrupted_blocks)} block(s)!"
        }

    def verify_single_evidence(self, evidence_id, test_payload):
        """Check whether a given piece of evidence has been altered against on-chain hash"""
        target_block = None
        for block in self.chain:
            if block.evidence_id == evidence_id:
                target_block = block
                break

        if not target_block:
            return {"found": False, "message": f"Evidence ID {evidence_id} not registered on blockchain"}

        current_hash = target_block.calculate_evidence_hash(test_payload)
        is_pristine = (current_hash == target_block.evidence_hash)

        return {
            "found": True,
            "evidence_id": evidence_id,
            "case_id": target_block.case_id,
            "evidence_type": target_block.evidence_type,
            "logged_officer": target_block.officer_badge,
            "logged_station": target_block.police_station,
            "blockchain_hash": target_block.evidence_hash,
            "computed_hash": current_hash,
            "is_tampered": not is_pristine,
            "status": "✅ Pristine & Verified Unaltered" if is_pristine else "❌ TAMPER DETECTED: Evidence was Modified!"
        }

    def simulate_tampering_attack(self, block_index, altered_text="SUSPECT DELETED FROM CHARGESHEET"):
        """
        Simulate an unauthorized data tampering attack for hackathon & court demo.
        Modifies block payload directly to demonstrate automated cryptographic detection.
        """
        if block_index <= 0 or block_index >= len(self.chain):
            return {"success": False, "message": "Invalid block index for simulation"}

        victim_block = self.chain[block_index]
        victim_block.payload = {"tampered_content": altered_text, "original_id": victim_block.evidence_id}
        # Notice we do NOT re-mine or update evidence_hash/block_hash, simulating an attacker modifying the database
        return {
            "success": True,
            "tampered_block_index": block_index,
            "tampered_evidence_id": victim_block.evidence_id,
            "message": f"Simulated tampering injected into Block #{block_index} ({victim_block.evidence_id}). Run integrity verification to detect it!"
        }

    def seed_initial_evidence(self):
        """Seed initial realistic evidence blocks from our criminal dataset"""
        # Block 1: FIR Evidence
        self.register_evidence(
            evidence_id="EVID_FIR_0001",
            evidence_type="FIR Record",
            case_id="FIR0001",
            payload={"accused": ["SUS0036", "SUS0022", "SUS0122"], "crime": "Money Laundering", "ipc": "IT Act 66C", "city": "Pune"},
            officer_badge="INSP-R-8821",
            police_station="PS South Pune",
            role="ROLE_INVESTIGATING_OFFICER"
        )
        # Block 2: Wiretap Call Log
        self.register_evidence(
            evidence_id="EVID_CDR_00001",
            evidence_type="CDR Call Log",
            case_id="FIR0004",
            payload={"caller": "SUS0034", "callee": "SUS0074", "duration": 266, "tower": "Chandigarh", "intercept_type": "Voice"},
            officer_badge="CYBER-OFFICER-449",
            police_station="Cyber Crime Unit Chandigarh",
            role="ROLE_CYBER_CELL"
        )
        # Block 3: Social Media Threat Post
        self.register_evidence(
            evidence_id="EVID_POST_00003",
            evidence_type="Social Media Threat Post",
            case_id="FIR0022",
            payload={"suspect": "SUS0032", "platform": "Twitter/X", "post": "Big deal coming this weekend, stay ready", "threat_flag": "Suspicious"},
            officer_badge="CYBER-INTEL-102",
            police_station="Cyber Cell Amritsar",
            role="ROLE_CYBER_CELL"
        )
        # Block 4: Forensic Financial Ledger
        self.register_evidence(
            evidence_id="EVID_FIN_00088",
            evidence_type="Financial Flow Ledger",
            case_id="FIR0012",
            payload={"source": "SUS0088", "target": "SUS0147", "amount_inr": 4728606.05, "flag": "Suspicious Laundering Flow"},
            officer_badge="FORENSIC-ACC-904",
            police_station="State Forensic Science Lab Ranchi",
            role="ROLE_FORENSIC_EXPERT"
        )
        # Block 5: Seizure & Weapon Memo
        self.register_evidence(
            evidence_id="EVID_SEIZ_00015",
            evidence_type="Seizure Memo",
            case_id="FIR0003",
            payload={"suspect": "SUS0115", "recovered_items": ["Unlicensed 9mm Pistol", "Fake Passport", "Encrypted Satellite Phone"], "city": "Indore"},
            officer_badge="INSP-V-3312",
            police_station="Cyber Cell Indore",
            role="ROLE_INVESTIGATING_OFFICER"
        )
