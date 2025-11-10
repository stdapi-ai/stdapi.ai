
# Getting Started with stdapi.ai

This guide will help you deploy stdapi.ai on AWS and make your first API call.

## Why stdapi.ai?

stdapi.ai provides OpenAI-compatible access to AWS Bedrock and other AI services with infrastructure costs starting at $7/month.

**Key benefits:**

- Deploy in minutes with pre-built Terraform modules
- Enterprise security with zero vendor lock-in—your data stays in your AWS account
- Production-ready monitoring, auto-scaling, and high availability
- Transparent pricing with fixed infrastructure costs
- Pre-configured best practices for AWS deployments

## What You'll Accomplish

By the end of this guide, you'll have:

- A deployed stdapi.ai instance running on AWS
- Successfully made your first API call using OpenAI SDKs
- Infrastructure ready for building AI-powered applications

## Deploy to Your AWS Account

### Choose Your Deployment Path

**New to stdapi.ai? Start here:**
- [**Quick Start (3 lines of code)**](#example-1-quick-start-3-lines-of-code) - Get running in 5 minutes

**Already have AWS infrastructure?**
- [**Integrate with Existing VPC/ALB**](#example-2-integration-with-existing-infrastructure) - Most cost-effective

**Need production features?**
- [**Production Deployment**](#example-3-production-deployment-fully-featured) - HTTPS, WAF, multi-region

**Looking for ultra-low cost?**
- [**Cost-Optimized Development**](#example-4-cost-optimized-deployment) - Scheduled Spot instances

**Don't use Terraform?**
- [**Manual ECS Deployment**](#option-b-manual-ecs-deployment) - Deploy the container directly

---

### Option A: Terraform Module (Recommended)

The Terraform module provides production-ready infrastructure with minimal configuration.

**Advantages:**

- Deploy with just 3 lines of code using sensible defaults
- HTTPS, monitoring, auto-scaling, and security hardening included
- CloudWatch alarms with anomaly detection and centralized logging
- Optional WAF with AWS managed rules and KMS encryption
- S3 lifecycle policies, auto-scaling controls, and Fargate Spot support
- Version-controlled, repeatable deployments
- Multi-region deployment support

#### Prerequisites

1. **Subscribe to stdapi.ai** on [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo) (required for container access)
2. Install [Terraform](https://www.terraform.io/downloads) or [OpenTofu](https://opentofu.org/docs/intro/install/) >= 1.5
3. Configure [AWS credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)

#### Example 1: Quick Start (3 lines of code)

The simplest deployment option. Ideal for first-time users, testing, or proof-of-concept.

```hcl
module "stdapi_ai" {
  source  = "stdapi-ai/stdapi-ai/aws"
  version = "~> 1.0"
}
```

**Deploy it:**

```bash
terraform init
terraform apply
```

Your API is now running. The endpoint URL will be shown in the Terraform outputs.

**What you get:**

- ECS Fargate service (0.25 vCPU, 512 MiB RAM)
- Secure VPC with private subnets and NAT gateways
- CloudWatch logging
- Internal service discovery

**How to access your API:**

After deployment, retrieve your endpoint:

```bash
terraform output
```

For external access, you have two options:

1. **Add a load balancer** (recommended for production):
   ```hcl
   module "stdapi_ai" {
     source  = "stdapi-ai/stdapi-ai/aws"
     version = "~> 1.0"

     alb_enabled = true  # Adds Application Load Balancer
   }
   ```

2. **Use with existing infrastructure** - See Example 2 below

**Monthly cost:** ~$45 (Fargate ~$7 + NAT Gateway ~$37 + KMS $1) + AI model usage + license

*Costs shown are for a single AZ deployment in the default configuration.*

**First deployment?** Start here, make your first API call, then explore other examples for production features or cost optimization.

For lower costs, see Example 2 (~$8/month with existing infrastructure) or Example 4 (~$2.50/month for development).

---

#### Example 2: Integration with Existing Infrastructure

Deploy stdapi.ai into your existing VPC and network infrastructure for maximum cost efficiency (~$8/month).

```hcl
module "stdapi_ai" {
  source  = "stdapi-ai/stdapi-ai/aws"
  version = "~> 1.0"

  # Use your existing network
  subnet_ids = [
    "subnet-xxx",  # Your existing private subnet 1
    "subnet-yyy",  # Your existing private subnet 2
  ]
  security_group_id = "sg-zzz"  # Your existing security group
}
```

**What you get:**

- Same ECS Fargate service as Example 1
- Integrates with your existing VPC and networking
- No additional NAT gateways or load balancers needed
- Full monitoring and security features

**How to connect to your ALB:**

After deployment, add a target group pointing to port 8000:

```hcl
resource "aws_lb_target_group" "stdapi" {
  name        = "stdapi-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = "vpc-xxxxx"
  target_type = "ip"

  health_check {
    path = "/health"
  }
}
```

See the full integration example in the collapsed section below for complete ALB configuration.

**Monthly cost:** ~$8 (Fargate ~$7 + KMS $1) + AI model usage + license

*This cost assumes you're reusing existing VPC, subnets, and load balancer infrastructure.*

**Best value:** 94% cost reduction vs standalone deployment by reusing existing infrastructure.

??? example "📋 Full integration example with ALB, IAM policies, and advanced configuration"

    **Complete integration configuration with all optional features:**

    ```hcl
    module "stdapi_ai_integrated" {
      source  = "stdapi-ai/stdapi-ai/aws"
      version = "~> 1.0"

      name_prefix = "my-stdapi-integrated"

      # Use existing network infrastructure
      subnet_ids = [
        "subnet-0123456789abcdef0",
        "subnet-0123456789abcdef1",
        "subnet-0123456789abcdef2"
      ]
      security_group_id = "sg-0123456789abcdef0"

      # Optional: Reuse existing S3 bucket
      aws_s3_bucket = "my-existing-s3-bucket"

      # Optional: Service Discovery for private communication
      service_discovery_dns_namespace_id = "ns-xxxxx"
      service_discovery_dns_name         = "stdapi"

      # Optional: Use existing Secrets Manager secret for API key
      api_key_secretsmanager_secret = "my-api-keys"
      api_key_secretsmanager_key    = "stdapi_key"

      # Optional: Attach custom IAM policies
      ecs_task_role_policy_arns = [
        aws_iam_policy.custom_s3_access.arn,
        aws_iam_policy.api_key_secrets_access.arn
      ]

      # Monitoring
      container_insight = "enhanced"
      alarms_enabled    = true
      sns_topic_arn     = "arn:aws:sns:us-east-1:123456789012:alerts"
    }

    # Example: Custom IAM policy for additional S3 bucket access
    data "aws_iam_policy_document" "custom_s3_access" {
      statement {
        sid    = "S3BucketAccess"
        effect = "Allow"
        actions = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        resources = ["arn:aws:s3:::my-existing-s3-bucket/*"]
      }

      statement {
        sid    = "KMSEncryptionForS3"
        effect = "Allow"
        actions = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        resources = ["arn:aws:kms:us-east-1:123456789012:key/your-s3-bucket-kms-key-id"]
        condition {
          test     = "StringEquals"
          variable = "kms:ViaService"
          values   = ["s3.us-east-1.amazonaws.com"]
        }
      }
    }

    resource "aws_iam_policy" "custom_s3_access" {
      name        = "stdapi-custom-s3-access"
      description = "Custom S3 access for stdapi.ai integration"
      policy      = data.aws_iam_policy_document.custom_s3_access.json
    }

    # Example: IAM policy for API key access from Secrets Manager
    # Required when using api_key_secretsmanager_secret parameter
    data "aws_iam_policy_document" "api_key_secrets_access" {
      statement {
        sid       = "SecretsManagerAccess"
        effect    = "Allow"
        actions   = ["secretsmanager:GetSecretValue"]
        resources = ["arn:aws:secretsmanager:us-east-1:123456789012:secret:my-api-keys-*"]
      }

      statement {
        sid       = "KMSDecryptionForSecretsManager"
        effect    = "Allow"
        actions   = ["kms:Decrypt"]
        resources = ["arn:aws:kms:us-east-1:123456789012:key/your-kms-key-id"]
        condition {
          test     = "StringEquals"
          variable = "kms:ViaService"
          values   = ["secretsmanager.us-east-1.amazonaws.com"]
        }
      }
    }

    resource "aws_iam_policy" "api_key_secrets_access" {
      name        = "stdapi-api-key-secrets-access"
      description = "Access to Secrets Manager for stdapi.ai API key"
      policy      = data.aws_iam_policy_document.api_key_secrets_access.json
    }

    # Alternative: IAM policy for API key access from SSM Parameter Store
    # Use this when using api_key_ssm_parameter instead of Secrets Manager
    data "aws_iam_policy_document" "api_key_ssm_access" {
      statement {
        sid       = "SSMParameterAccess"
        effect    = "Allow"
        actions   = ["ssm:GetParameter"]
        resources = ["arn:aws:ssm:us-east-1:123456789012:parameter/stdapi/api-key"]
      }

      statement {
        sid       = "KMSDecryptionForSSM"
        effect    = "Allow"
        actions   = ["kms:Decrypt"]
        resources = ["arn:aws:kms:us-east-1:123456789012:key/your-kms-key-id"]
        condition {
          test     = "StringEquals"
          variable = "kms:ViaService"
          values   = ["ssm.us-east-1.amazonaws.com"]
        }
      }
    }

    resource "aws_iam_policy" "api_key_ssm_access" {
      name        = "stdapi-api-key-ssm-access"
      description = "Access to SSM Parameter Store for stdapi.ai API key"
      policy      = data.aws_iam_policy_document.api_key_ssm_access.json
    }

    # Outputs for integration
    output "ecs_service_info" {
      description = "ECS service details for connecting your resources"
      value       = {
        cluster_name      = module.stdapi_ai_integrated.ecs_cluster_name
        service_name      = module.stdapi_ai_integrated.ecs_service_name
        security_group_id = module.stdapi_ai_integrated.security_group_id
        port              = module.stdapi_ai_integrated.port
        service_discovery = module.stdapi_ai_integrated.service_discovery_service_name
      }
    }

    output "integration_resources" {
      description = "Resources for connecting stdapi.ai to your infrastructure"
      value       = {
        s3_bucket_id = module.stdapi_ai_integrated.bucket_id
        kms_key_arn  = module.stdapi_ai_integrated.kms_key_arn
        log_groups   = module.stdapi_ai_integrated.cloudwatch_log_groups_names
      }
    }
    ```

    **Manual integration steps:**

    1. **Configure your ALB target group** to point to the ECS service:
       ```hcl
       resource "aws_lb_target_group" "stdapi" {
         name        = "my-stdapi-tg"
         port        = 8000
         protocol    = "HTTP"
         vpc_id      = "vpc-xxxxx"
         target_type = "ip"

         health_check {
           path                = "/health"
           healthy_threshold   = 2
           unhealthy_threshold = 3
         }
       }

       # Attach to your existing ALB listener
       resource "aws_lb_listener_rule" "stdapi" {
         listener_arn = aws_lb_listener.existing.arn
         priority     = 100

         action {
           type             = "forward"
           target_group_arn = aws_lb_target_group.stdapi.arn
         }

         condition {
           path_pattern {
             values = ["/v1/*"]
           }
         }
       }
       ```

    2. **Update security groups** to allow traffic:
       ```hcl
       # Allow your ALB to reach stdapi.ai
       resource "aws_security_group_rule" "alb_to_stdapi" {
         type                     = "ingress"
         from_port                = 8000
         to_port                  = 8000
         protocol                 = "tcp"
         security_group_id        = module.stdapi_ai_integrated.security_group_id
         source_security_group_id = var.your_alb_security_group_id
       }
       ```

    3. **Access via Service Discovery** (optional):
       ```python
       # If service discovery is enabled, access via private DNS
       client = OpenAI(
           api_key="YOUR_API_KEY",
           base_url="http://stdapi.your-namespace.local:8000/v1"
       )
       ```

    **Use cases:**

    - Connect to existing internal ALB
    - Private API for internal microservices
    - Connect to service mesh (App Mesh, Consul)
    - Custom networking with VPN/Direct Connect
    - Multi-account setups with PrivateLink
    - Access additional AWS resources (S3 buckets, Secrets Manager, DynamoDB, etc.)

    **Custom IAM policies use cases:**

    - Grant access to additional S3 buckets beyond the default one
    - **Access API keys from Secrets Manager or SSM Parameter Store** (required when using `api_key_ssm_parameter` or `api_key_secretsmanager_secret`)
    - Read/write to DynamoDB tables for application state
    - Access to custom KMS keys for encryption
    - Cross-account resource access via IAM roles

    **Important:** When using `api_key_secretsmanager_secret` or `api_key_ssm_parameter`, you must create and attach an IAM policy granting the ECS task access to the secret/parameter. The module does not automatically create these permissions.

---

#### Example 3: Production Deployment (Fully Featured)

Enterprise-ready deployment with HTTPS endpoints, WAF protection, auto-scaling, and comprehensive monitoring.

**About regional S3 buckets:** The full production example below includes multi-region Bedrock support with regional S3 buckets. This is only required for Bedrock multimodal features (images, documents) across multiple regions. For most users, the simplified single-region setup is sufficient—see the example further down.

??? example "📋 Full production example with multi-region Bedrock support"

    ```hcl
    # Main deployment
    module "stdapi_ai" {
      source  = "stdapi-ai/stdapi-ai/aws"
      version = "~> 1.0"
    
      # Custom public domain with TLS
      alb_domain_name   = "api.example.com"
      alb_enabled       = true
      alb_public        = true
    
      # AWS Bedrock region configuration
      # Select regions to get available models in the order of preference
      aws_bedrock_regions = [
        "eu-west-3",
        "eu-west-1",
        "eu-central-1",
        "eu-north-1"
      ]
      
      # Regional buckets for Bedrock multimodal operations
      # Required by some models and features, create one per extra region in aws_bedrock_regions
      aws_s3_regional_buckets = merge(
        module.bedrock_bucket_eu_west_1.regional_bucket_map,
        module.bedrock_bucket_eu_central_1.regional_bucket_map,
        module.bedrock_bucket_eu_north_1.regional_bucket_map,
      )
      aws_s3_buckets_kms_keys_arns = [
        module.bedrock_bucket_eu_west_1.kms_key_arn,
        module.bedrock_bucket_eu_central_1.kms_key_arn,
        module.bedrock_bucket_eu_north_1.kms_key_arn,
      ]
      
      # (Optional) In case of regional compliance requirements like GDPR,
      # disable "global" cross-region inference to ensure everything is done in valid regions.
      # Cross-region inference allows AWS Bedrock to route requests to different regions for better availability.
      # In this example, cross-region inferences will be in EU regions only and comply with GDPR
      aws_bedrock_cross_region_inference_global = false
    
      # AI services region extra configuration
      # Required if a service or a feature is not available in your main region
      # In this example, AWS Comprehend is not available on eu-west-3, so we use eu-west-1
      aws_comprehend_region = "eu-west-1" 
    
      # Authentication (Recommended)
      # Enable authentication by generating an API key that can be retrieved using the "api_key" module attribute.
      api_key_create = true
    
      # Web Application Firewall (Recommended on public APIs when ALB is enabled)
      alb_waf_enabled             = true
      alb_waf_rate_limit          = 2000  # Requests per 5 minutes per IP
      alb_waf_block_anonymous_ips = true
    
      # Monitoring & Alerts (Recommended to get alarms notifications)
      alarms_enabled = true
      sns_topic_arn  = "arn:aws:sns:eu-west-3:123456789012:alerts"
    }
    
    # Get the API key (Generated with api_key_create = true)
    
    output "api_key" {
      value     = module.stdapi_ai.api_key
      sensitive = true
    }
    
    # Main/default region provider
    provider "aws" {
      region = "eu-west-3"
    }
    
    # Additional providers for regional Bedrock buckets
    
    provider "aws" {
      alias  = "eu-central-1"
      region = "eu-central-1"
    }
    
    provider "aws" {
      alias  = "eu-west-1"
      region = "eu-west-1"
    }
    
    provider "aws" {
      alias  = "eu-north-1"
      region = "eu-north-1"
    }
    
    # Regional S3 buckets for Bedrock operations (optional but recommended)
    module "bedrock_bucket_eu_west_1" {
      source  = "stdapi-ai/stdapi-ai-s3-regional-bucket/aws"
      version = "~> 1.0"
    
      providers = { aws = aws.eu-west-1 }
      name_prefix = module.stdapi_ai.name_prefix
      aws_s3_tmp_prefix = module.stdapi_ai.aws_s3_tmp_prefix
      deletion_protection = module.stdapi_ai.deletion_protection
    }
    
    module "bedrock_bucket_eu_central_1" {
      source  = "stdapi-ai/stdapi-ai-s3-regional-bucket/aws"
      version = "~> 1.0"
    
      providers = { aws = aws.eu-central-1 }
      name_prefix = module.stdapi_ai.name_prefix
      aws_s3_tmp_prefix = module.stdapi_ai.aws_s3_tmp_prefix
      deletion_protection = module.stdapi_ai.deletion_protection
    }
    
    module "bedrock_bucket_eu_north_1" {
      source  = "stdapi-ai/stdapi-ai-s3-regional-bucket/aws"
      version = "~> 1.0"
    
      providers = { aws = aws.eu-north-1 }
      name_prefix = module.stdapi_ai.name_prefix
      aws_s3_tmp_prefix = module.stdapi_ai.aws_s3_tmp_prefix
      deletion_protection = module.stdapi_ai.deletion_protection
    }
    
    ```

**What you get:**

- High-availability multi-AZ deployment (uses all available AZs in region)
- HTTPS with automatic SSL certificate
- WAF protection with AWS managed rules
- 5 CloudWatch alarms (memory, health, CPU anomaly, capacity, error logs)
- Auto-scaling 2-10 tasks based on load
- S3 storage with lifecycle policies
- Enhanced Container Insights
- Regional S3 buckets for Bedrock multimodal operations in 3 regions

**Estimated monthly cost (us-east-1):**

- Fargate (0.25 vCPU ARM64, 512 MiB, 730 hours): $7.21
- NAT Gateway + EIP: $36.50
- ALB (fixed + LCU charges): $16.43
- ACM Certificate: $1.00
- KMS key: $1.00
- WAF: ~$5.00
- **Total:** ~$67/month (single AZ) + AI model usage + license

*Multi-AZ deployments incur additional Fargate and NAT Gateway costs per AZ.*

**Included features:**

- HTTPS with automatic SSL certificates via ACM
- WAF protection with AWS managed rules and rate limiting
- Auto-scaling based on CPU, memory, and request count
- CloudWatch alarms for monitoring
- Multi-region support for AWS Bedrock operations
- Pre-configured AWS best practices

**Cost optimization:** Deploy into an existing VPC with existing ALB to reduce costs to ~$8/month. See Example 2.

**Deployment time:** ~5-10 minutes

??? example "📋 Simplified production example (single region, no multi-region complexity)"

    ```hcl
    module "stdapi_ai" {
      source  = "stdapi-ai/stdapi-ai/aws"
      version = "~> 1.0"
    
      # HTTPS with your domain
      alb_domain_name = "api.example.com"
      alb_enabled     = true
      alb_public      = true
    
      # Security
      api_key_create              = true
      alb_waf_enabled             = true
      alb_waf_rate_limit          = 2000
      alb_waf_block_anonymous_ips = true
    
      # Monitoring
      alarms_enabled = true
      sns_topic_arn  = "arn:aws:sns:us-east-1:123456789012:alerts"
    }
    
    output "api_key" {
      value     = module.stdapi_ai.api_key
      sensitive = true
    }
    
    output "api_endpoint" {
      value = module.stdapi_ai.application_url
    }
    ```

**Deploy and get your API key:**

```bash
terraform init
terraform apply
terraform output -raw api_key  # Copy this for API calls
```

**Monthly cost:** ~$68 (Fargate ~$7 + NAT ~$37 + ALB ~$17 + WAF ~$5 + KMS $1 + Certificate $1) + AI model usage + license

Production-grade infrastructure with HTTPS, WAF protection, and monitoring without multi-region complexity.

---

#### Example 4: Cost-Optimized Deployment

Low-cost deployment for development and non-critical workloads (~$2.50/month). Suitable for side projects and development environments.


??? example "📋 Low cost deployment"

    ```hcl
    module "stdapi_ai_cost_optimized" {
      source  = "stdapi-ai/stdapi-ai/aws"
      version = "~> 1.0"
    
      # Aggressive Auto-scaling with Fargate spot
      autoscaling_enabled            = true
      autoscaling_min_capacity       = 1
      autoscaling_max_capacity       = 3
      autoscaling_cpu_target_percent = 85
      autoscaling_scale_in_cooldown  = 60   # Scale down quickly
      autoscaling_scale_out_cooldown = 120
      autoscaling_spot_percent       = 100  # Use 100% Spot pricing (~70% discount)
    
      # Schedule: Stop at 7 PM, start at 8 AM on weekdays (UTC)
      autoscaling_schedule_stop  = "cron(0 19 ? * MON-FRI *)"
      autoscaling_schedule_start = "cron(0 8 ? * MON-FRI *)"
    
      # Use Existing Subnets and security group (no VPC creation)
      subnet_ids = [
        "subnet-0123456789abcdef0",  # Your existing private subnet 1
        "subnet-0123456789abcdef1",  # Your existing private subnet 2
        "subnet-0123456789abcdef2"   # Your existing private subnet 3
      ]
      security_group_id = "sg-0123456789abcdef0"
    
      # Minimal Monitoring & Logging
      container_insight                 = "disabled"  # Disable Container Insights
      vpc_flow_log_enabled              = false       # Disable VPC Flow Logs
      cloudwatch_logs_retention_in_days = 7           # Reduce log retention to 7 days
    }
    ```

**What you get:**

- Fargate Spot for 70% cost reduction
- Default minimal resources (0.25 vCPU ARM64, 512 MiB)
- Reuse existing VPC infrastructure
- Automated scheduling (runs 8 AM-7 PM weekdays only in UTC)
- Minimal logging (7-day retention, no Container Insights, no VPC Flow Logs)
- S3 Intelligent-Tiering for storage optimization

**Estimated monthly cost (us-east-1):**

- Fargate Spot (0.25 vCPU ARM64, 512 MiB, ~55 hours/week): ~$1.50
- KMS key: $1
- **Total:** ~$2.50/month (scales to ~$4/month with 3 tasks during peak) + AI model usage + license

**Trade-offs:** Spot interruptions possible, minimal observability (7-day logs, no Container Insights or VPC Flow Logs)

---

#### Outputs

After deployment, access critical information:

```hcl
output "api_endpoint" {
  value = module.stdapi_ai_prod.alb_dns_name
}
```

**Available outputs for all deployment scenarios:**

**Networking & Load Balancing:**

- `alb_dns_name` - ALB endpoint (if enabled)
- `alb_arn` - ALB ARN for AWS integrations
- `alb_security_group_id` - ALB security group
- `alb_target_group_arn` - Target group for custom listeners
- `application_url` - Full URL (https://domain or http://alb)

**ECS Service:**

- `ecs_cluster_name` - Cluster name for AWS CLI/SDK
- `ecs_service_name` - Service name for management
- `security_group_id` - Security group for ingress rules
- `service_discovery_service_name` - Private DNS name (if enabled)
- `port` - Container port exposed by the application

**Storage & Encryption:**

- `bucket_id` - S3 bucket for application data
- `bucket_arn` - S3 bucket ARN
- `kms_key_id` - KMS key for encryption
- `kms_key_arn` - KMS ARN for IAM policies

**Security:**

- `waf_web_acl_id` - WAF ACL ID (if enabled)
- `waf_web_acl_arn` - WAF ACL ARN (if enabled)

For the complete list of outputs, see [stdapi-ai/terraform-aws-stdapi-ai/outputs.tf](https://github.com/stdapi-ai/terraform-aws-stdapi-ai/outputs.tf).

---

### Option B: Manual ECS Deployment

If you prefer not to use Terraform or need a custom deployment, you can deploy the stdapi.ai container image directly to AWS ECS after subscribing to the AWS Marketplace listing.

#### Prerequisites

1. **Subscribe to stdapi.ai** on [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo) (required for container access)
2. Set up an ECS cluster (Fargate or EC2)
3. Configure networking (VPC, subnets, security groups)
4. Set up IAM roles with appropriate permissions

#### Container Image

After subscribing, the container image is available from AWS Marketplace ECR:

```
709825985650.dkr.ecr.us-east-1.amazonaws.com/j-goutin/stdapi.ai:<version>
```

#### ECS Task Definition Example

```json
{
  "family": "stdapi-ai-task-definition",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::{account-id}:role/{execution-role-name}",
  "taskRoleArn": "arn:aws:iam::{account-id}:role/{task-role-name}",
  "runtimePlatform": {
    "cpuArchitecture": "ARM64",
    "operatingSystemFamily": "LINUX"
  },
  "containerDefinitions": [
    {
      "name": "main",
      "image": "709825985650.dkr.ecr.us-east-1.amazonaws.com/j-goutin/stdapi.ai:1.0.1-arm64",
      "essential": true,
      "readonlyRootFilesystem": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp",
          "name": "http"
        }
      ],
      "environment": [
        {
          "name": "AWS_S3_BUCKET",
          "value": "{your-s3-bucket-name}"
        },
        {
          "name": "AWS_BEDROCK_REGIONS",
          "value": "us-east-1,us-west-2"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "temp",
          "containerPath": "/tmp"
        }
      ],
      "healthCheck": {
        "command": [
          "CMD",
          "python3",
          "-c",
          "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 30
      },
      "linuxParameters": {
        "capabilities": {
          "drop": ["ALL"]
        }
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/stdapi-ai",
          "awslogs-region": "{region}",
          "awslogs-stream-prefix": "stdapi-ai"
        }
      }
    }
  ],
  "volumes": [
    {
      "name": "temp"
    }
  ]
}
```

**Note:** This is a minimal example. For production, configure:

- Environment variables (see [Configuration](operations_configuration.md))
- Health checks
- IAM task roles for AWS service access
- Load balancer integration
- Auto-scaling policies
- CloudWatch monitoring

**Recommendation:** Use the Terraform module (Option A) for a complete, production-ready deployment with all best practices included.

## Make Your First API Call

stdapi.ai is OpenAI-compatible. If you've used OpenAI before, you already know how to use it.

### Step 1: Get your endpoint and API key

```bash
# Get your endpoint URL
terraform output application_url

# Get your API key (if you enabled api_key_create=true)
terraform output -raw api_key
```

### Step 2: Make your first call

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key-here",           # From terraform output
    base_url="https://your-url-here/v1"    # From terraform output
)

# That's it! Use it exactly like OpenAI
response = client.chat.completions.create(
    model="anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=[{"role": "user", "content": "Hello! Tell me a joke."}]
)

print(response.choices[0].message.content)
```

**No API key configured?** By default, stdapi.ai runs without authentication for quick testing. Add `api_key_create = true` to your Terraform config to enable authentication.

**Available models:** All AWS Bedrock models available in your region are supported.

You're now running your own AI infrastructure on AWS.

## Next Steps

- [API overview](api_overview.md) – Available endpoints and usage
- [Configuration](operations_configuration.md) – Customize your deployment
- [Licensing](operations_licensing.md) – Dual licensing options (AGPL and commercial)
- [Roadmap](roadmap.md) – Upcoming features and compatibility

## Troubleshooting

### VPC Endpoint Error: "couldn't find resource" for AWS Comprehend

**Error message:**

```
Error: reading EC2 VPC Endpoint Services: couldn't find resource

  with module.stdapi_ai.module.vpc.data.aws_vpc_endpoint_service.netdev_vpce_interface["comprehend"],
  on module-stdapi-ai/module-vpc/network_devices.tf line 175, in data "aws_vpc_endpoint_service" "netdev_vpce_interface":
 175: data "aws_vpc_endpoint_service" "netdev_vpce_interface" {
```

**Cause:** AWS Comprehend is not available as a VPC endpoint service in your current region. Not all AWS services have VPC endpoints in all regions.

**Solution:** Set the `aws_comprehend_region` variable in your Terraform module configuration to specify a region where Comprehend is available:

```hcl
module "stdapi_ai" {
  source  = "stdapi-ai/stdapi-ai/aws"
  version = "~> 1.0"

  name_prefix = "my-stdapi"

  # Set Comprehend to use a different region
  aws_comprehend_region = "us-east-1"  # or another region where Comprehend is available
}
```

Common regions with Comprehend support: `us-east-1`, `us-west-2`, `eu-west-1`, `eu-central-1`
