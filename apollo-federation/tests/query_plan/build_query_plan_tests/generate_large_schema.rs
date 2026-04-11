/// Generates large, complex federated schemas with realistic patterns and measures
/// shared condition cache performance.
///
/// Run with: USE_ROVER=1 cargo test -p apollo-federation -- generate_large_schema --nocapture
use apollo_federation::query_plan::{PlanNode, QueryPlan, TopLevelPlanNode};
use std::fmt::Write as FmtWrite;
use std::time::Instant;

/// Structural stats for a query plan — how complex is the plan actually?
#[derive(Default, Debug, Clone, Copy)]
struct PlanShape {
    fetches: usize,
    parallel_groups: usize,
    sequence_groups: usize,
    flatten_nodes: usize,
    condition_nodes: usize,
    defer_nodes: usize,
    max_depth: usize,
    // Max number of children in any single Parallel node (true fan-out width)
    max_parallel_fan_out: usize,
    // Distinct subgraphs fetched from
    distinct_subgraphs: usize,
}

fn analyze_plan(plan: &QueryPlan) -> PlanShape {
    let mut shape = PlanShape::default();
    let mut subgraphs: std::collections::BTreeSet<String> = Default::default();
    if let Some(node) = &plan.node {
        walk_top(node, 1, &mut shape, &mut subgraphs);
    }
    shape.distinct_subgraphs = subgraphs.len();
    shape
}

fn walk_top(
    node: &TopLevelPlanNode,
    depth: usize,
    shape: &mut PlanShape,
    subgraphs: &mut std::collections::BTreeSet<String>,
) {
    shape.max_depth = shape.max_depth.max(depth);
    match node {
        TopLevelPlanNode::Fetch(f) => {
            shape.fetches += 1;
            subgraphs.insert(f.subgraph_name.to_string());
        }
        TopLevelPlanNode::Sequence(s) => {
            shape.sequence_groups += 1;
            for n in &s.nodes {
                walk(n, depth + 1, shape, subgraphs);
            }
        }
        TopLevelPlanNode::Parallel(p) => {
            shape.parallel_groups += 1;
            shape.max_parallel_fan_out = shape.max_parallel_fan_out.max(p.nodes.len());
            for n in &p.nodes {
                walk(n, depth + 1, shape, subgraphs);
            }
        }
        TopLevelPlanNode::Flatten(f) => {
            shape.flatten_nodes += 1;
            walk(&f.node, depth + 1, shape, subgraphs);
        }
        TopLevelPlanNode::Defer(d) => {
            shape.defer_nodes += 1;
            if let Some(n) = d.primary.node.as_deref() {
                walk(n, depth + 1, shape, subgraphs);
            }
            for db in &d.deferred {
                if let Some(n) = db.node.as_deref() {
                    walk(n, depth + 1, shape, subgraphs);
                }
            }
        }
        TopLevelPlanNode::Condition(c) => {
            shape.condition_nodes += 1;
            if let Some(n) = c.if_clause.as_deref() {
                walk(n, depth + 1, shape, subgraphs);
            }
            if let Some(n) = c.else_clause.as_deref() {
                walk(n, depth + 1, shape, subgraphs);
            }
        }
        TopLevelPlanNode::Subscription(s) => {
            shape.fetches += 1;
            subgraphs.insert(s.primary.subgraph_name.to_string());
            if let Some(n) = s.rest.as_deref() {
                walk(n, depth + 1, shape, subgraphs);
            }
        }
    }
}

fn walk(
    node: &PlanNode,
    depth: usize,
    shape: &mut PlanShape,
    subgraphs: &mut std::collections::BTreeSet<String>,
) {
    shape.max_depth = shape.max_depth.max(depth);
    match node {
        PlanNode::Fetch(f) => {
            shape.fetches += 1;
            subgraphs.insert(f.subgraph_name.to_string());
        }
        PlanNode::Sequence(s) => {
            shape.sequence_groups += 1;
            for n in &s.nodes {
                walk(n, depth + 1, shape, subgraphs);
            }
        }
        PlanNode::Parallel(p) => {
            shape.parallel_groups += 1;
            shape.max_parallel_fan_out = shape.max_parallel_fan_out.max(p.nodes.len());
            for n in &p.nodes {
                walk(n, depth + 1, shape, subgraphs);
            }
        }
        PlanNode::Flatten(f) => {
            shape.flatten_nodes += 1;
            walk(&f.node, depth + 1, shape, subgraphs);
        }
        PlanNode::Defer(d) => {
            shape.defer_nodes += 1;
            if let Some(n) = d.primary.node.as_deref() {
                walk(n, depth + 1, shape, subgraphs);
            }
            for db in &d.deferred {
                if let Some(n) = db.node.as_deref() {
                    walk(n, depth + 1, shape, subgraphs);
                }
            }
        }
        PlanNode::Condition(c) => {
            shape.condition_nodes += 1;
            if let Some(n) = c.if_clause.as_deref() {
                walk(n, depth + 1, shape, subgraphs);
            }
            if let Some(n) = c.else_clause.as_deref() {
                walk(n, depth + 1, shape, subgraphs);
            }
        }
    }
}

const LINK: &str = r#"@link(url: "https://specs.apollo.dev/federation/v2.9", import: ["@key", "@requires", "@provides", "@external", "@tag", "@extends", "@shareable", "@inaccessible", "@override", "@composeDirective", "@interfaceObject", "@context", "@fromContext", "@cost", "@listSize"])"#;

/// Builds a realistic e-commerce/social platform supergraph.
///
/// Design goals:
/// - Shared entity types resolved across many subgraphs (User, Product, Order)
/// - Multiple @key definitions per type (id, slug, sku, email)
/// - Deep @requires chains (A requires B requires C requires D)
/// - @provides for pre-fetched cross-subgraph data
/// - Interface types implemented across subgraphs
/// - Union types spanning subgraphs
/// - Diamond dependency patterns (same entity reachable via multiple paths)
/// - Deeply nested object types (3-5 levels)
/// - Varying subgraph complexity (some tiny, some large)
fn generate_subgraphs(scale: usize) -> Vec<(String, String)> {
    let mut subgraphs = Vec::new();

    // ============================================================
    // CORE SUBGRAPHS — the backbone entity owners
    // ============================================================

    // --- users: owns User, Account, UserProfile ---
    subgraphs.push(("users".into(), format!(
        r#"extend schema {LINK}

type Query {{
  me: User
  user(id: ID!): User
  userByEmail(email: String!): User
  users(limit: Int = 10, offset: Int = 0): [User!]!
}}

type Mutation {{
  updateUser(id: ID!, input: UpdateUserInput!): User!
}}

input UpdateUserInput {{
  name: String
  bio: String
}}

type User @key(fields: "id") @key(fields: "email") @key(fields: "username") {{
  id: ID!
  email: String!
  username: String!
  name: String!
  createdAt: String!
  role: UserRole!
  profile: UserProfile!
  settings: UserSettings!
  addresses: [Address!]!
}}

type UserProfile {{
  bio: String
  avatarUrl: String
  location: Location
  website: String
  socialLinks: SocialLinks
}}

type SocialLinks {{
  twitter: String
  github: String
  linkedin: String
}}

type Location {{
  city: String
  state: String
  country: String!
  coordinates: Coordinates
}}

type Coordinates {{
  lat: Float!
  lng: Float!
}}

type UserSettings {{
  emailNotifications: Boolean!
  theme: String!
  language: String!
  timezone: String!
}}

type Address @key(fields: "id") {{
  id: ID!
  label: String!
  street: String!
  city: String!
  state: String!
  zip: String!
  country: String!
  isDefault: Boolean!
}}

enum UserRole {{
  ADMIN
  CUSTOMER
  VENDOR
  SUPPORT
}}

type Account @key(fields: "id") @key(fields: "accountNumber") {{
  id: ID!
  accountNumber: String!
  owner: User!
  balance: Float!
  currency: String!
  tier: AccountTier!
  openedAt: String!
  limits: AccountLimits!
}}

type AccountLimits {{
  dailyTransfer: Float!
  monthlyTransfer: Float!
  singleTransaction: Float!
}}

enum AccountTier {{
  FREE
  BASIC
  PREMIUM
  ENTERPRISE
}}
"#)));

    // --- products: owns Product, Category, Brand with deep nesting ---
    subgraphs.push(("products".into(), format!(
        r#"extend schema {LINK}

type Query {{
  product(upc: String!): Product
  productBySku(sku: String!): Product
  productBySlug(slug: String!): Product
  products(first: Int = 10, after: String, filter: ProductFilter): ProductConnection!
  categories: [Category!]!
  category(id: ID!): Category
  brands: [Brand!]!
  topProducts(first: Int = 5): [Product!]!
}}

input ProductFilter {{
  categoryId: ID
  brandId: ID
  minPrice: Float
  maxPrice: Float
  inStock: Boolean
  tags: [String!]
}}

type Product @key(fields: "upc") @key(fields: "sku") @key(fields: "slug") {{
  upc: String!
  sku: String!
  slug: String!
  name: String!
  price: Float!
  weight: Float!
  dimensions: ProductDimensions!
  category: Category!
  brand: Brand!
  description: String
  tags: [String!]!
  variants: [ProductVariant!]!
  images: [ProductImage!]!
  attributes: [ProductAttribute!]!
  createdAt: String!
  updatedAt: String!
}}

type ProductDimensions {{
  length: Float!
  width: Float!
  height: Float!
  unit: DimensionUnit!
}}

enum DimensionUnit {{
  CM
  IN
}}

type ProductVariant @key(fields: "id") {{
  id: ID!
  name: String!
  sku: String!
  price: Float!
  color: String
  size: String
  weight: Float
}}

type ProductImage {{
  url: String!
  alt: String
  width: Int!
  height: Int!
  isPrimary: Boolean!
}}

type ProductAttribute {{
  key: String!
  value: String!
}}

type Category @key(fields: "id") @key(fields: "slug") {{
  id: ID!
  slug: String!
  name: String!
  description: String
  parent: Category
  children: [Category!]!
  breadcrumb: [Category!]!
  depth: Int!
  productCount: Int!
}}

type Brand @key(fields: "id") @key(fields: "slug") {{
  id: ID!
  slug: String!
  name: String!
  description: String
  logoUrl: String
  country: String
  foundedYear: Int
  website: String
}}

type ProductConnection {{
  edges: [ProductEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}}

type ProductEdge {{
  node: Product!
  cursor: String!
}}

type PageInfo {{
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}}
"#)));

    // --- inventory: extends Product with stock, @requires price+weight ---
    subgraphs.push(("inventory".into(), format!(
        r#"extend schema {LINK}

type Query {{
  warehouseStock(warehouseId: ID!): [InventoryEntry!]!
}}

type Product @key(fields: "upc") {{
  upc: String!
  price: Float! @external
  weight: Float! @external
  name: String! @external
  inStock: Boolean!
  stockCount: Int!
  warehouses: [WarehouseStock!]!
  inventoryValue: Float! @requires(fields: "price")
  shippingEstimate: ShippingEstimate! @requires(fields: "weight price")
  inventoryLabel: String! @requires(fields: "name")
  restockDate: String
  lowStockAlert: Boolean!
}}

type WarehouseStock {{
  warehouseId: ID!
  warehouseName: String!
  quantity: Int!
  location: WarehouseLocation!
  lastRestocked: String!
}}

type WarehouseLocation {{
  region: String!
  zone: String!
  aisle: String!
  shelf: String!
}}

type ShippingEstimate {{
  standardDays: Int!
  expressDays: Int!
  standardCost: Float!
  expressCost: Float!
}}

type InventoryEntry {{
  product: Product!
  quantity: Int!
  reservedQuantity: Int!
  availableQuantity: Int!
}}

type ProductVariant @key(fields: "id") {{
  id: ID!
  price: Float! @external
  variantStock: Int!
  variantInventoryValue: Float! @requires(fields: "price")
}}
"#)));

    // --- reviews: extends Product and User, owns Review ---
    // Uses @provides to send author name along with review data
    subgraphs.push(("reviews".into(), format!(
        r#"extend schema {LINK}

type Query {{
  reviews(productUpc: String!): [Review!]!
  latestReviews(limit: Int = 10): [Review!]!
  review(id: ID!): Review
}}

type Review @key(fields: "id") {{
  id: ID!
  title: String!
  body: String!
  rating: Int!
  author: User!
  product: Product!
  createdAt: String!
  updatedAt: String
  helpfulVotes: Int!
  verifiedPurchase: Boolean!
  media: [ReviewMedia!]!
  response: SellerResponse
}}

type ReviewMedia {{
  url: String!
  type: MediaType!
  caption: String
}}

enum MediaType {{
  IMAGE
  VIDEO
}}

type SellerResponse {{
  body: String!
  respondedAt: String!
}}

type User @key(fields: "id") {{
  id: ID!
  reviews: [Review!]!
  reviewCount: Int!
  averageRating: Float!
  reviewDistribution: ReviewDistribution!
}}

type ReviewDistribution {{
  oneStar: Int!
  twoStar: Int!
  threeStar: Int!
  fourStar: Int!
  fiveStar: Int!
}}

type Product @key(fields: "upc") {{
  upc: String!
  reviews: [Review!]!
  reviewCount: Int!
  averageRating: Float!
  reviewSummary: ReviewSummary!
}}

type ReviewSummary {{
  totalReviews: Int!
  averageRating: Float!
  ratingDistribution: ReviewDistribution!
  topPositiveReview: Review
  topCriticalReview: Review
}}
"#)));

    // --- orders: extends User/Product, deep nesting, @requires ---
    subgraphs.push(("orders".into(), format!(
        r#"extend schema {LINK}

type Query {{
  order(id: ID!): Order
  myOrders(limit: Int = 10, status: OrderStatus): [Order!]!
}}

type Mutation {{
  placeOrder(input: PlaceOrderInput!): Order!
  cancelOrder(id: ID!): Order!
}}

input PlaceOrderInput {{
  items: [OrderItemInput!]!
  shippingAddressId: ID!
  paymentMethodId: ID!
}}

input OrderItemInput {{
  productUpc: String!
  quantity: Int!
}}

type Order @key(fields: "id") @key(fields: "orderNumber") {{
  id: ID!
  orderNumber: String!
  customer: User!
  items: [OrderItem!]!
  status: OrderStatus!
  totalAmount: Float!
  subtotal: Float!
  taxAmount: Float!
  placedAt: String!
  updatedAt: String!
  shippingAddress: OrderAddress!
  billingAddress: OrderAddress!
  payment: PaymentInfo!
  timeline: [OrderEvent!]!
}}

type OrderItem @key(fields: "id") {{
  id: ID!
  product: Product!
  quantity: Int!
  unitPrice: Float!
  totalPrice: Float!
  customization: OrderItemCustomization
}}

type OrderItemCustomization {{
  color: String
  size: String
  engraving: String
  giftWrap: Boolean!
  giftMessage: String
}}

type OrderAddress {{
  name: String!
  street: String!
  city: String!
  state: String!
  zip: String!
  country: String!
  phone: String
}}

type PaymentInfo {{
  method: PaymentMethod!
  last4: String!
  transactionId: String!
}}

enum PaymentMethod {{
  CREDIT_CARD
  DEBIT_CARD
  PAYPAL
  APPLE_PAY
  CRYPTO
}}

type OrderEvent {{
  type: OrderEventType!
  timestamp: String!
  description: String!
}}

enum OrderEventType {{
  PLACED
  CONFIRMED
  PROCESSING
  SHIPPED
  DELIVERED
  CANCELLED
  REFUNDED
}}

enum OrderStatus {{
  PENDING
  CONFIRMED
  PROCESSING
  SHIPPED
  DELIVERED
  CANCELLED
  REFUNDED
}}

type User @key(fields: "id") {{
  id: ID!
  name: String! @external
  orders: [Order!]!
  totalSpent: Float!
  orderCount: Int!
  lastOrderDate: String
  loyaltyTier: LoyaltyTier! @requires(fields: "name")
}}

enum LoyaltyTier {{
  BRONZE
  SILVER
  GOLD
  PLATINUM
  DIAMOND
}}

type Product @key(fields: "upc") {{
  upc: String!
  price: Float! @external
  orderCount: Int!
  totalRevenue: Float! @requires(fields: "price")
}}
"#)));

    // --- shipping: extends Order with shipment tracking, deep @requires ---
    subgraphs.push(("shipping".into(), format!(
        r#"extend schema {LINK}

type Query {{
  shipment(trackingNumber: String!): Shipment
  estimateShipping(productUpc: String!, destination: String!): ShippingQuote!
}}

type Order @key(fields: "id") {{
  id: ID!
  totalAmount: Float! @external
  shipments: [Shipment!]!
  estimatedDelivery: String!
  shippingCost: Float! @requires(fields: "totalAmount")
}}

type Shipment @key(fields: "trackingNumber") {{
  trackingNumber: String!
  carrier: Carrier!
  status: ShipmentStatus!
  estimatedArrival: String!
  actualArrival: String
  weight: Float!
  packages: [Package!]!
  trackingEvents: [TrackingEvent!]!
}}

type Carrier {{
  name: String!
  code: String!
  trackingUrl: String!
}}

type Package {{
  packageNumber: Int!
  weight: Float!
  dimensions: PackageDimensions!
  items: [PackageItem!]!
}}

type PackageDimensions {{
  length: Float!
  width: Float!
  height: Float!
}}

type PackageItem {{
  description: String!
  quantity: Int!
}}

type TrackingEvent {{
  status: String!
  location: String!
  timestamp: String!
  description: String!
}}

enum ShipmentStatus {{
  LABEL_CREATED
  PICKED_UP
  IN_TRANSIT
  OUT_FOR_DELIVERY
  DELIVERED
  EXCEPTION
  RETURNED
}}

type ShippingQuote {{
  standard: ShippingOption!
  express: ShippingOption!
  overnight: ShippingOption
}}

type ShippingOption {{
  cost: Float!
  estimatedDays: Int!
  carrier: String!
}}
"#)));

    // --- analytics: extends many types with computed fields, deep @requires chains ---
    subgraphs.push(("analytics".into(), format!(
        r#"extend schema {LINK}

type Product @key(fields: "upc") {{
  upc: String!
  price: Float! @external
  name: String! @external
  weight: Float! @external
  salesRank: Int!
  viewCount: Int!
  conversionRate: Float!
  pricePerformanceScore: Float! @requires(fields: "price")
  searchRelevanceScore: Float! @requires(fields: "name")
  shippingEfficiency: Float! @requires(fields: "weight price")
  trendDirection: TrendDirection!
  competitorCount: Int!
}}

enum TrendDirection {{
  UP
  DOWN
  STABLE
}}

type User @key(fields: "id") {{
  id: ID!
  name: String! @external
  email: String! @external
  engagementScore: Float!
  lifetimeValue: Float!
  churnRisk: Float! @requires(fields: "name")
  segmentId: String! @requires(fields: "email")
  cohort: UserCohort!
}}

type UserCohort {{
  name: String!
  joinedAt: String!
  size: Int!
}}

type Category @key(fields: "id") {{
  id: ID!
  name: String! @external
  trendingScore: Float!
  categoryPerformance: Float! @requires(fields: "name")
  seasonalIndex: Float!
  growthRate: Float!
}}

type Brand @key(fields: "id") {{
  id: ID!
  name: String! @external
  brandStrength: Float! @requires(fields: "name")
  marketShare: Float!
  sentimentScore: Float!
  npsScore: Int!
}}

type Review @key(fields: "id") {{
  id: ID!
  rating: Int! @external
  body: String! @external
  sentimentScore: Float! @requires(fields: "body")
  qualityScore: Float! @requires(fields: "rating body")
  isSpam: Boolean!
}}

type Order @key(fields: "id") {{
  id: ID!
  totalAmount: Float! @external
  profitMargin: Float! @requires(fields: "totalAmount")
  fraudRiskScore: Float!
}}
"#)));

    // --- pricing: extends Product with dynamic pricing, deep requires ---
    subgraphs.push(("pricing".into(), format!(
        r#"extend schema {LINK}

type Product @key(fields: "upc") {{
  upc: String!
  price: Float! @external
  name: String! @external
  weight: Float! @external
  dynamicPrice: Float! @requires(fields: "price")
  discount: Discount
  priceHistory: [PricePoint!]!
  competitorPriceIndex: Float! @requires(fields: "price name")
  bundleDiscount: Float! @requires(fields: "price weight")
  priceAlert: PriceAlert
}}

type Discount {{
  percentage: Float!
  absoluteAmount: Float!
  validUntil: String!
  code: String
  minimumQuantity: Int
}}

type PricePoint {{
  price: Float!
  date: String!
  source: PriceSource!
}}

enum PriceSource {{
  MANUAL
  ALGORITHM
  COMPETITOR_MATCH
  PROMOTION
}}

type PriceAlert {{
  targetPrice: Float!
  currentPrice: Float!
  direction: PriceAlertDirection!
}}

enum PriceAlertDirection {{
  ABOVE
  BELOW
}}

type ProductVariant @key(fields: "id") {{
  id: ID!
  price: Float! @external
  variantDynamicPrice: Float! @requires(fields: "price")
  variantDiscount: Discount
  variantPriceHistory: [PricePoint!]!
}}
"#)));

    // --- recommendations: extends User/Product, @requires user history ---
    subgraphs.push(("recommendations".into(), format!(
        r#"extend schema {LINK}

type User @key(fields: "id") {{
  id: ID!
  name: String! @external
  email: String! @external
  recommendedProducts: [Product!]! @requires(fields: "name")
  similarUsers: [User!]!
  personalizedFeed: [FeedItem!]! @requires(fields: "name email")
  browsingHistory: [BrowsingEvent!]!
}}

type FeedItem {{
  type: FeedItemType!
  product: Product
  content: String!
  score: Float!
  reason: String!
}}

enum FeedItemType {{
  PRODUCT_RECOMMENDATION
  DEAL_ALERT
  TRENDING
  BASED_ON_HISTORY
}}

type BrowsingEvent {{
  productUpc: String!
  viewedAt: String!
  durationSeconds: Int!
}}

type Product @key(fields: "upc") {{
  upc: String!
  name: String! @external
  price: Float! @external
  relatedProducts: [Product!]!
  frequentlyBoughtWith: [Product!]!
  personalizedRank: Float! @requires(fields: "name")
  dealScore: Float! @requires(fields: "price name")
}}
"#)));

    // --- search: union across entity types ---
    subgraphs.push(("search".into(), format!(
        r#"extend schema {LINK}

type Query {{
  search(query: String!, limit: Int = 10, types: [SearchType!]): SearchResultConnection!
  autocomplete(prefix: String!, limit: Int = 5): [AutocompleteResult!]!
}}

enum SearchType {{
  PRODUCT
  USER
  BRAND
  CATEGORY
  ORDER
}}

type SearchResultConnection {{
  results: [SearchResult!]!
  totalCount: Int!
  facets: [SearchFacet!]!
}}

union SearchResult = ProductSearchHit | UserSearchHit | BrandSearchHit | CategorySearchHit

type ProductSearchHit {{
  product: Product!
  relevanceScore: Float!
  highlights: [String!]!
}}

type UserSearchHit {{
  user: User!
  relevanceScore: Float!
}}

type BrandSearchHit {{
  brand: Brand!
  relevanceScore: Float!
}}

type CategorySearchHit {{
  category: Category!
  relevanceScore: Float!
}}

type SearchFacet {{
  field: String!
  values: [FacetValue!]!
}}

type FacetValue {{
  value: String!
  count: Int!
}}

type AutocompleteResult {{
  text: String!
  type: SearchType!
  entityId: String!
}}

type Product @key(fields: "upc") {{
  upc: String!
}}

type User @key(fields: "id") {{
  id: ID!
}}

type Brand @key(fields: "id") {{
  id: ID!
}}

type Category @key(fields: "id") {{
  id: ID!
}}
"#)));

    // --- payments: extends Order/User/Account with payment processing ---
    subgraphs.push(("payments".into(), format!(
        r#"extend schema {LINK}

type Query {{
  paymentMethods: [StoredPaymentMethod!]!
}}

type Mutation {{
  addPaymentMethod(input: AddPaymentMethodInput!): StoredPaymentMethod!
}}

input AddPaymentMethodInput {{
  type: String!
  token: String!
  isDefault: Boolean
}}

type User @key(fields: "id") {{
  id: ID!
  paymentMethods: [StoredPaymentMethod!]!
  defaultPaymentMethod: StoredPaymentMethod
}}

type StoredPaymentMethod @key(fields: "id") {{
  id: ID!
  type: String!
  last4: String!
  expiryMonth: Int
  expiryYear: Int
  isDefault: Boolean!
  billingAddress: PaymentAddress
}}

type PaymentAddress {{
  street: String!
  city: String!
  state: String!
  zip: String!
  country: String!
}}

type Order @key(fields: "id") {{
  id: ID!
  totalAmount: Float! @external
  paymentStatus: PaymentStatus!
  paymentTransactions: [PaymentTransaction!]!
  refundEligible: Boolean! @requires(fields: "totalAmount")
}}

type PaymentTransaction {{
  id: ID!
  amount: Float!
  status: PaymentStatus!
  processedAt: String!
  method: String!
  gatewayReference: String
}}

enum PaymentStatus {{
  PENDING
  AUTHORIZED
  CAPTURED
  REFUNDED
  FAILED
  DISPUTED
}}

type Account @key(fields: "id") {{
  id: ID!
  balance: Float! @external
  currency: String! @external
  paymentHistory: [PaymentTransaction!]!
  availableCredit: Float! @requires(fields: "balance currency")
}}
"#)));

    // --- notifications: extends User/Order with notification preferences ---
    subgraphs.push(("notifications".into(), format!(
        r#"extend schema {LINK}

type User @key(fields: "id") {{
  id: ID!
  email: String! @external
  name: String! @external
  notificationPreferences: NotificationPreferences!
  unreadNotificationCount: Int!
  notifications(limit: Int = 20): [Notification!]!
  notificationDigest: NotificationDigest! @requires(fields: "email name")
}}

type NotificationPreferences {{
  email: Boolean!
  push: Boolean!
  sms: Boolean!
  orderUpdates: Boolean!
  promotions: Boolean!
  reviewResponses: Boolean!
}}

type Notification {{
  id: ID!
  type: NotificationType!
  title: String!
  body: String!
  read: Boolean!
  createdAt: String!
  actionUrl: String
}}

enum NotificationType {{
  ORDER_UPDATE
  SHIPPING_UPDATE
  REVIEW_RESPONSE
  PRICE_DROP
  BACK_IN_STOCK
  PROMOTION
}}

type NotificationDigest {{
  dailySummary: Boolean!
  weeklySummary: Boolean!
  preferredTime: String!
}}

type Order @key(fields: "id") {{
  id: ID!
  status: OrderStatus! @external
  notificationsSent: [OrderNotification!]!
  nextNotification: String @requires(fields: "status")
}}

enum OrderStatus {{
  PENDING
  CONFIRMED
  PROCESSING
  SHIPPED
  DELIVERED
  CANCELLED
  REFUNDED
}}

type OrderNotification {{
  type: NotificationType!
  sentAt: String!
  channel: String!
}}
"#)));

    // ============================================================
    // SHAREABLE ZONE — three mirror subgraphs expose the same entity
    // with @shareable fields. Queries selecting multiple of these
    // fields force cost-based plan comparison because each field can
    // be sourced from any of the three subgraphs, yielding 3^N plan
    // candidates (capped by the planner's max_evaluated_plans budget).
    // ============================================================
    const SHAREABLE_BODY: &str = r#"
type Query {
  shareableRoot: ShareableEntity @shareable
  shareableList(limit: Int = 10): [ShareableEntity!]! @shareable
}

type ShareableEntity @key(fields: "id") @shareable {
  id: ID!
  fieldA: String!
  fieldB: String!
  fieldC: String!
  fieldD: String!
  fieldE: String!
  fieldF: String!
  fieldG: String!
  fieldH: String!
  nested: ShareableNested!
}

type ShareableNested @shareable {
  inner1: String!
  inner2: String!
  inner3: String!
}
"#;
    for mirror in ["alpha", "beta", "gamma"] {
        subgraphs.push((
            format!("shareable_{mirror}"),
            format!("extend schema {LINK}\n{SHAREABLE_BODY}"),
        ));
    }

    // ============================================================
    // SCALED DEPARTMENT SUBGRAPHS — each adds types + cross-refs
    // ============================================================
    for i in 0..scale {
        let mut schema = format!("extend schema {LINK}\n\n");

        // Vary the complexity per department
        let has_deep_requires = i % 3 == 0;
        let has_provides = i % 2 == 0;
        let has_interface = i % 5 == 0;
        let has_multiple_keys = i % 4 == 0;
        let depth = (i % 3) + 2; // nesting depth 2-4

        // Department-specific types with varying nesting depth
        if has_interface {
            write!(schema, r#"
interface Dept{i}Content {{
  id: ID!
  title: String!
  createdAt: String!
}}

type Dept{i}Article implements Dept{i}Content @key(fields: "id") {{
  id: ID!
  title: String!
  createdAt: String!
  body: String!
  author: User!
  tags: [String!]!
}}

type Dept{i}Guide implements Dept{i}Content @key(fields: "id") {{
  id: ID!
  title: String!
  createdAt: String!
  steps: [Dept{i}GuideStep!]!
  difficulty: String!
}}

type Dept{i}GuideStep {{
  stepNumber: Int!
  instruction: String!
  imageUrl: String
}}

"#).unwrap();
        }

        // Build nested metadata types to the specified depth
        let mut nested_type_name = format!("Dept{i}Leaf");
        write!(schema, r#"
type {nested_type_name} {{
  value: String!
  confidence: Float!
  source: String!
  timestamp: String!
}}
"#).unwrap();

        for d in (0..depth).rev() {
            let parent_name = if d == 0 {
                format!("Dept{i}Metadata")
            } else {
                format!("Dept{i}Level{d}")
            };
            write!(schema, r#"
type {parent_name} {{
  label: String!
  data: {nested_type_name}!
  priority: Int!
  tags: [String!]!
}}
"#).unwrap();
            nested_type_name = parent_name;
        }

        // Main department item type
        let key_directive = if has_multiple_keys {
            format!(r#"@key(fields: "id") @key(fields: "code")"#)
        } else {
            r#"@key(fields: "id")"#.to_string()
        };

        write!(schema, r#"
type Query {{
  dept{i}Items(limit: Int = 10, filter: Dept{i}Filter): [Dept{i}Item!]!
  dept{i}Item(id: ID!): Dept{i}Item
  dept{i}Stats: Dept{i}Stats!
}}

input Dept{i}Filter {{
  minPrice: Float
  maxPrice: Float
  available: Boolean
  tags: [String!]
}}

type Dept{i}Item {key_directive} {{
  id: ID!{code_field}
  name: String!
  product: Product!
  price: Float!
  dept{i}Rating: Float!
  metadata: Dept{i}Metadata!
  status: Dept{i}Status!
  suppliers: [Dept{i}Supplier!]!
}}

enum Dept{i}Status {{
  ACTIVE
  DISCONTINUED
  SEASONAL
  PREORDER
}}

type Dept{i}Supplier {{
  name: String!
  leadTimeDays: Int!
  cost: Float!
  reliability: Float!
}}

type Dept{i}Stats {{
  totalItems: Int!
  averagePrice: Float!
  topItem: Dept{i}Item
}}
"#,
            code_field = if has_multiple_keys { format!("\n  code: String!") } else { String::new() },
        ).unwrap();

        // Staff type with user cross-ref
        write!(schema, r#"
type Dept{i}Staff @key(fields: "employeeId") {{
  employeeId: ID!
  name: String!
  role: String!
  user: User
  hireDate: String!
  department: String!
}}
"#).unwrap();

        // Product extension with @requires and optionally @provides
        if has_provides {
            write!(schema, r#"
type Product @key(fields: "upc") {{
  upc: String!
  name: String! @external
  price: Float! @external
  weight: Float! @external
  dept{i}Available: Boolean!
  dept{i}Price: Float!
  dept{i}Discount: Float! @requires(fields: "name price")
  dept{i}ShippingClass: String! @requires(fields: "weight")
}}
"#).unwrap();
        } else {
            write!(schema, r#"
type Product @key(fields: "upc") {{
  upc: String!
  name: String! @external
  price: Float! @external
  dept{i}Available: Boolean!
  dept{i}Price: Float!
  dept{i}Discount: Float! @requires(fields: "name price")
}}
"#).unwrap();
        }

        // User extension with @requires
        if has_deep_requires {
            // Deep requires chain: dept field requires name, which requires...
            write!(schema, r#"
type User @key(fields: "id") {{
  id: ID!
  name: String! @external
  email: String! @external
  dept{i}Preferences: Dept{i}UserPrefs! @requires(fields: "name email")
  dept{i}Score: Float! @requires(fields: "name")
}}

type Dept{i}UserPrefs {{
  favoriteCategories: [String!]!
  priceRange: Dept{i}PriceRange!
  notifications: Boolean!
}}

type Dept{i}PriceRange {{
  min: Float!
  max: Float!
}}
"#).unwrap();
        } else {
            write!(schema, r#"
type User @key(fields: "id") {{
  id: ID!
  name: String! @external
  dept{i}Preferences: [String!]! @requires(fields: "name")
}}
"#).unwrap();
        }

        // Cross-department references for some departments
        if i > 0 && i % 3 == 0 {
            let prev = i - 1;
            write!(schema, r#"
type Dept{prev}Item @key(fields: "id") {{
  id: ID!
  dept{i}CrossReference: String
}}
"#).unwrap();
        }

        subgraphs.push((format!("dept{i}"), schema));
    }

    subgraphs
}

/// Builds a query that asks for many department-contributed Product fields at once.
/// Each `dept{i}Discount` carries `@requires(fields: "name price")` and each
/// `dept{i}ShippingClass` (on even depts) carries `@requires(fields: "weight")`, so
/// this forces the planner to reason about many parallel `@requires` sites sharing
/// a common `name/price/weight` prefetch.
fn stress_product_cross_depts(num_depts: usize) -> String {
    let mut q = String::from("{ topProducts { upc name price weight salesRank dynamicPrice ");
    for i in 0..num_depts {
        write!(
            q,
            "dept{i}Available dept{i}Price dept{i}Discount "
        )
        .unwrap();
        // Even-indexed depts also have @requires(weight) via dept{i}ShippingClass
        if i % 2 == 0 {
            write!(q, "dept{i}ShippingClass ").unwrap();
        }
    }
    q.push_str("} }");
    q
}

/// Builds a query that reaches into many department-contributed User fields at once.
/// Every 3rd department has `dept{i}Preferences: Dept{i}UserPrefs @requires(name email)`
/// and `dept{i}Score @requires(name)`; the rest have `dept{i}Preferences: [String!]!
/// @requires(name)`. Cross-cutting them forces the planner to resolve many parallel
/// `@requires` sites on the same User entity.
fn stress_user_cross_depts(num_depts: usize) -> String {
    let mut q = String::from("{ me { id name email role ");
    for i in 0..num_depts {
        if i % 3 == 0 {
            // Object-returning preferences + score
            write!(
                q,
                "dept{i}Preferences {{ favoriteCategories notifications priceRange {{ min max }} }} dept{i}Score "
            )
            .unwrap();
        } else {
            // Scalar-list preferences only
            write!(q, "dept{i}Preferences ").unwrap();
        }
    }
    q.push_str("} }");
    q
}

/// Cross-subgraph diamond: single query pulls data for the same top-level entity
/// set (topProducts) through analytics, pricing, inventory, reviews, AND dept{0..N}
/// extensions simultaneously. Combined with the many-department pressure this is
/// the worst-case single plan we can generate with this schema.
fn stress_mega_diamond(num_depts: usize) -> String {
    let mut q = String::from(
        "{ topProducts { \
         upc name price weight \
         dimensions { length width height unit } \
         brand { name country foundedYear } \
         category { name parent { name parent { name } } } \
         inStock stockCount inventoryValue shippingEstimate { standardDays standardCost expressDays expressCost } \
         reviews { rating title body author { id name } } \
         reviewCount averageRating \
         salesRank viewCount conversionRate pricePerformanceScore trendDirection \
         dynamicPrice competitorPriceIndex bundleDiscount discount { percentage absoluteAmount } \
         priceHistory { price date source } \
         ",
    );
    for i in 0..num_depts {
        write!(
            q,
            "dept{i}Available dept{i}Price dept{i}Discount "
        )
        .unwrap();
        if i % 2 == 0 {
            write!(q, "dept{i}ShippingClass ").unwrap();
        }
    }
    q.push_str("} }");
    q
}

/// Stress query that selects N shareable fields from the shareable entity.
/// Each field is resolvable from 3 mirror subgraphs, so the planner has
/// 3^N candidate plans to evaluate (bounded by max_evaluated_plans = 10,000).
fn stress_shareable_multi_plan(num_fields: usize) -> String {
    let all_fields = [
        "fieldA", "fieldB", "fieldC", "fieldD", "fieldE", "fieldF", "fieldG", "fieldH",
    ];
    let n = num_fields.min(all_fields.len());
    let mut q = String::from("{ shareableRoot { id");
    for f in &all_fields[..n] {
        write!(q, " {f}").unwrap();
    }
    q.push_str(" } }");
    q
}

/// Same but pulls from the list-returning shareable root — the list form
/// may produce different plan enumeration because it goes through an
/// entities jump rather than a scalar root.
fn stress_shareable_list_multi_plan(num_fields: usize) -> String {
    let all_fields = [
        "fieldA", "fieldB", "fieldC", "fieldD", "fieldE", "fieldF", "fieldG", "fieldH",
    ];
    let n = num_fields.min(all_fields.len());
    let mut q = String::from("{ shareableList { id");
    for f in &all_fields[..n] {
        write!(q, " {f}").unwrap();
    }
    q.push_str(" nested { inner1 inner2 inner3 } } }");
    q
}

fn compose_with_rover(subgraphs: &[(String, String)]) -> String {
    let temp_dir = tempfile::tempdir().unwrap();
    let temp_path = temp_dir.path();

    let mut config = String::from("federation_version: =2.9.0\nsubgraphs:\n");
    for (name, schema) in subgraphs {
        let subgraph_path = temp_path.join(format!("{name}.graphql"));
        std::fs::write(&subgraph_path, schema).unwrap();
        write!(
            config,
            "  {name}:\n    routing_url: none\n    schema:\n      file: {}\n",
            subgraph_path.display()
        )
        .unwrap();
    }

    let config_path = temp_path.join("rover.yaml");
    std::fs::write(&config_path, &config).unwrap();

    let output = std::process::Command::new("rover")
        .args(["supergraph", "compose", "--config"])
        .arg(&config_path)
        .output()
        .expect("rover must be in PATH");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("rover compose failed:\n{stderr}");
    }

    String::from_utf8(output.stdout).unwrap()
}

fn plan_and_measure(
    label: &str,
    supergraph_sdl: &str,
    queries: &[&str],
    iterations: usize,
    dump_plan_for: Option<usize>,
) {
    let supergraph =
        apollo_federation::Supergraph::new(supergraph_sdl).expect("supergraph should be valid");
    let api_schema = supergraph
        .to_api_schema(apollo_federation::ApiSchemaOptions::default())
        .expect("api schema should be valid");
    let planner = apollo_federation::query_plan::query_planner::QueryPlanner::new(
        &supergraph,
        Default::default(),
    )
    .expect("planner should be created");

    eprintln!("\n  === {label} ===");
    eprintln!(
        "    {:<56} {:>9} {:>9} {:>7} {:>7} {:>6} {:>4} {:>4} {:>4} {:>5} {:>6} {:>20}",
        "query",
        "cold-µs",
        "warm-µs",
        "speed",
        "paths",
        "plans",
        "ftc",
        "sub",
        "dep",
        "par/s",
        "fanOt",
        "warm norm/cmp/proc µs",
    );

    for (idx, query_str) in queries.iter().enumerate() {
        let doc = match apollo_compiler::ExecutableDocument::parse_and_validate(
            api_schema.schema(),
            *query_str,
            "bench.graphql",
        ) {
            Ok(d) => d,
            Err(e) => {
                let q_short = if query_str.len() > 60 {
                    format!("{}...", &query_str[..57])
                } else {
                    query_str.to_string()
                };
                eprintln!("    SKIP `{q_short}`: {e}");
                continue;
            }
        };

        // Each iteration: (wall, paths, plans, shape, normalize_ns, compute_ns, process_ns)
        let mut timings: Vec<(
            std::time::Duration,
            usize,
            usize,
            PlanShape,
            u128,
            u128,
            u128,
        )> = Vec::new();
        let mut last_plan_str: Option<String> = None;
        for i in 0..iterations {
            let start = Instant::now();
            let plan = planner
                .build_query_plan(
                    &doc,
                    None,
                    apollo_federation::query_plan::query_planner::QueryPlanOptions::default(),
                )
                .expect("plan should succeed");
            let elapsed = start.elapsed();
            let shape = analyze_plan(&plan);
            let norm_ns = plan.statistics.phase_timings.normalize_ns.get();
            let comp_ns = plan.statistics.phase_timings.compute_dep_graph_ns.get();
            let proc_ns = plan.statistics.phase_timings.dep_graph_process_ns.get();
            timings.push((
                elapsed,
                plan.statistics.evaluated_plan_paths.get(),
                plan.statistics.evaluated_plan_count.get(),
                shape,
                norm_ns,
                comp_ns,
                proc_ns,
            ));
            if Some(idx) == dump_plan_for && i == 0 {
                last_plan_str = Some(format!("{plan}"));
            }
        }

        let cold = timings[0].0;
        let cold_paths = timings[0].1;
        let cold_plans = timings[0].2;
        let shape = timings[0].3;
        let warm_count = (timings.len() - 1) as u32;
        let warm_avg: std::time::Duration = timings[1..]
            .iter()
            .map(|(d, _, _, _, _, _, _)| *d)
            .sum::<std::time::Duration>()
            / warm_count;
        let warm_norm_avg_us: u128 = timings[1..]
            .iter()
            .map(|(_, _, _, _, n, _, _)| *n)
            .sum::<u128>()
            / warm_count as u128
            / 1_000;
        let warm_comp_avg_us: u128 = timings[1..]
            .iter()
            .map(|(_, _, _, _, _, c, _)| *c)
            .sum::<u128>()
            / warm_count as u128
            / 1_000;
        let warm_proc_avg_us: u128 = timings[1..]
            .iter()
            .map(|(_, _, _, _, _, _, p)| *p)
            .sum::<u128>()
            / warm_count as u128
            / 1_000;
        let speedup = cold.as_nanos() as f64 / warm_avg.as_nanos() as f64;

        let q_short = if query_str.len() > 54 {
            format!("{}...", &query_str[..51])
        } else {
            query_str.to_string()
        };
        eprintln!(
            "    {:<56} {:>9} {:>9} {:>6.2}x {:>7} {:>6} {:>4} {:>4} {:>4} {:>5} {:>6} {:>20}",
            q_short,
            cold.as_micros(),
            warm_avg.as_micros(),
            speedup,
            cold_paths,
            cold_plans,
            shape.fetches,
            shape.distinct_subgraphs,
            shape.max_depth,
            format!("{}/{}", shape.parallel_groups, shape.sequence_groups),
            shape.max_parallel_fan_out,
            format!(
                "{:>5}/{:>5}/{:>5}",
                warm_norm_avg_us, warm_comp_avg_us, warm_proc_avg_us
            ),
        );

        if let Some(plan_str) = last_plan_str {
            eprintln!("\n    -- sample plan for query #{idx} --");
            for line in plan_str.lines().take(120) {
                eprintln!("    {line}");
            }
            if plan_str.lines().count() > 120 {
                eprintln!(
                    "    ... ({} more lines truncated)",
                    plan_str.lines().count() - 120
                );
            }
            eprintln!();
        }
    }
    eprintln!(
        "    cache: {} entries",
        planner.condition_resolver_cache_len()
    );
    eprintln!(
        "    legend: ftc=fetch nodes, sub=distinct subgraphs, dep=max depth, par/s=parallel/sequence groups, fanOt=max parallel children"
    );
    eprintln!(
        "            norm/cmp/proc = warm-avg phase timings in µs (normalize / compute-dep-graph / process-dep-graph)"
    );
}

#[test]
fn measure_large_schema_cache_performance() {
    if std::env::var_os("USE_ROVER").is_none() {
        eprintln!("SKIP: set USE_ROVER=1 to run this test");
        return;
    }

    let iterations = 10;

    // Queries that exercise different patterns
    let deep_cross_subgraph = r#"{ me { name email profile { bio location { city country coordinates { lat lng } } socialLinks { twitter github } } reviews { title rating body product { name price weight dimensions { length width height } brand { name country } category { name parent { name } } inStock inventoryValue shippingEstimate { standardDays standardCost expressCost } salesRank pricePerformanceScore dynamicPrice discount { percentage validUntil } } } } }"#;
    let wide_touch_many = r#"{ me { name orders { orderNumber status totalAmount items { product { name price } quantity } shippingAddress { city country } payment { method last4 } timeline { type timestamp } } reviews { rating title } recommendedProducts { name price } notificationPreferences { email push orderUpdates } paymentMethods { type last4 isDefault } } }"#;
    let analytics_requires_chain = r#"{ topProducts { name price weight salesRank viewCount conversionRate pricePerformanceScore searchRelevanceScore shippingEfficiency trendDirection dynamicPrice competitorPriceIndex bundleDiscount discount { percentage absoluteAmount } priceHistory { price date source } } }"#;
    let search_union = r#"{ search(query: "laptop", limit: 5) { results { ... on ProductSearchHit { product { upc name price } relevanceScore highlights } ... on UserSearchHit { user { id name } } ... on BrandSearchHit { brand { name country } } ... on CategorySearchHit { category { name depth } } } totalCount facets { field values { value count } } } }"#;
    let shipping_deep_requires = r#"{ order(id: "1") { orderNumber status totalAmount items { product { name price inStock } quantity unitPrice customization { color size giftWrap } } shipments { trackingNumber carrier { name trackingUrl } status packages { weight dimensions { length width height } items { description quantity } } trackingEvents { status location timestamp } } shippingCost estimatedDelivery payment { method last4 transactionId } } }"#;
    // dept0 has nesting depth 2 — Dept0Metadata.data is Dept0Level1, Dept0Level1.data is Dept0Leaf
    let dept_cross_refs = r#"{ dept0Items { name price metadata { label priority tags data { label priority data { value confidence source timestamp } } } product { upc name dept0Discount dept0ShippingClass dept0Available dept0Price } suppliers { name leadTimeDays cost reliability } } }"#;
    let user_full_profile = r#"{ me { name email username role profile { bio avatarUrl location { city state country coordinates { lat lng } } website socialLinks { twitter github linkedin } } settings { emailNotifications theme language timezone } addresses { label street city state zip country isDefault } engagementScore lifetimeValue churnRisk segmentId cohort { name size } loyaltyTier recommendedProducts { name price relatedProducts { name } } personalizedFeed { type content score reason } browsingHistory { productUpc viewedAt durationSeconds } notificationDigest { dailySummary weeklySummary preferredTime } } }"#;

    let base_queries: &[&str] = &[
        deep_cross_subgraph,
        wide_touch_many,
        analytics_requires_chain,
        search_union,
        shipping_deep_requires,
        dept_cross_refs,
        user_full_profile,
        // Repeat first query — should be fully warm
        deep_cross_subgraph,
    ];

    eprintln!("\n--- Large Schema Cache Performance ---");

    for num_depts in [10usize, 30, 50] {
        let subgraphs = generate_subgraphs(num_depts);
        eprintln!("\n  Composing {}-subgraph schema...", subgraphs.len());
        let start = Instant::now();
        let supergraph = compose_with_rover(&subgraphs);
        eprintln!(
            "  Composed in {:?} ({} bytes, {} lines)",
            start.elapsed(),
            supergraph.len(),
            supergraph.lines().count()
        );

        // @requires-site stress: cross-cut many dept extensions on a single entity
        let product_stress = stress_product_cross_depts(num_depts);
        let user_stress = stress_user_cross_depts(num_depts);
        let mega = stress_mega_diamond(num_depts);

        // Multi-plan stress: 3 shareable mirrors × N leaf fields → 3^N candidate plans.
        // At N=8 the planner would consider 6561 plans, right below its 10k budget.
        let shareable_small = stress_shareable_multi_plan(4); // 3^4 = 81
        let shareable_medium = stress_shareable_multi_plan(6); // 3^6 = 729
        let shareable_large = stress_shareable_multi_plan(8); // 3^8 = 6561
        let shareable_list = stress_shareable_list_multi_plan(6);

        let mut queries: Vec<&str> = base_queries.to_vec();
        queries.push(&product_stress);
        queries.push(&user_stress);
        queries.push(&mega);
        // Repeat mega to measure warm fan-out
        queries.push(&mega);
        queries.push(&shareable_small);
        queries.push(&shareable_medium);
        queries.push(&shareable_large);
        queries.push(&shareable_list);

        plan_and_measure(
            &format!("{} subgraphs (13 core + {} dept + 3 shareable)", subgraphs.len(), num_depts),
            &supergraph,
            &queries,
            iterations,
            // Dump the shareable_large plan (index 14) — shows multi-plan cost selection
            if num_depts == 50 { Some(14) } else { None },
        );
    }

    eprintln!("\n--- Done ---\n");
}
